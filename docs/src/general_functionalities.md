# General functionalities

Besides the sampler and the executor, BigHOPs.jl has a few features that make long or expensive searches easier to run: objectives that carry state between trials, stopping a run cleanly, checkpointing and resuming, and extending a finished run.

## Stateful objectives

A plain objective takes one `NamedTuple` of hyperparameters and returns the number to minimize. Wrapping it in `Stateful` lets it also receive and return state:

```julia
function objective(p; pre_artefact=nothing)
    model, epochs_done = pre_artefact === nothing ? (new_model(p), 0) : pre_artefact
    train!(model, p.r - epochs_done)
    return validation_loss(model), (model, p.r)
end

ho = Hyperoptimizer(Stateful(objective), candidates, Hyperband(R=81, η=3, r_min=1))
```

A `Stateful` objective is called as `f(p; pre_artefact)` and returns a tuple `(value, post_artefact)`. `value` is minimized as usual, and both artefacts can be anything.

### pre_artefact

`pre_artefact` is the state a trial starts from. It is `nothing`, except when a successive-halving sampler promotes a trial to a higher resource level: `Hyperband`, `ASHA` and `BOHB` then pass the earlier trial's `post_artefact` as the promoted trial's `pre_artefact`, so training continues where it stopped instead of paying again for the resource already spent. The objective has to handle both cases, as in the example above.

`Hyperband`, `ASHA` and `BOHB` warn when the objective is not `Stateful`, since every promotion then starts over. `DEHB` trains every trial from scratch and never sets `pre_artefact`, so it warns when the objective is `Stateful`.

Under `Serial` and `Threaded`, `pre_artefact` is the very object the earlier trial returned, so changing it in place, as `train!` does above, also changes what is stored for the earlier trial. Copy it first if the earlier state should be kept. Under `DistributedQueue`, each worker receives its own copy.

### post_artefact

Whatever a `Stateful` objective returns as `post_artefact` is stored with its trial, whichever sampler is used. With a full-budget sampler, this is a way to keep something from every trial, such as the trained model or its training curve, without saving it from inside the objective. The artefacts are columns of the trials' `DataFrame`:

```julia
using DataFrames

df = sort(dropmissing(DataFrame(ho), :value), :value)
best_model, _ = first(df).post_artefact
```

Every `post_artefact` is kept in memory for the whole run and written to every checkpoint, so it should be small enough to keep one per trial, e.g. a model's parameters rather than a copy of its training data.

## Trial states

Every trial is in one of five states, shown in the `status` column of `DataFrame(ho)`:

- `Pending`: the trial has been handed to the executor and has not finished yet.
- `Completed`: the objective returned a real number, which is the trial's `value`.
- `Failed`: the objective threw an exception, or returned `NaN` or something other than a number. Under `DistributedQueue`, a trial also fails when `spawn_worker` or `setup_worker` throws for it.
- `TimedOut`: under `DistributedQueue`, the trial ran longer than `max_single_runtime`, and its worker was killed.
- `Abandoned`: the trial was still running when its run stopped early, or when the checkpoint it was loaded from was written, so its result was never recorded.

A failing trial does not stop the search under any executor; it is recorded, and the run continues. For a `Failed` or `TimedOut` trial, the `error` column holds what went wrong: the exception, or whatever the objective returned instead of a number. Its hyperparameters stay in the results too, so it can be inspected and reproduced.

Only completed trials have a value, so `minimum`, `minimizer`, `history` and `results` use only them, but every trial counts towards the target `n`, whatever its state. The states are not exported, so filtering by them needs the module name:

```julia
df = DataFrame(ho)
failed = filter(:status => ==(BigHOPs.Failed), df)
```

## Graceful termination

A run can be stopped at any time with Ctrl-C. `run!` then stops starting new trials and shuts the executor down:

- under `Serial`, the trial that was interrupted is the only one running;
- under `Threaded`, it waits for the trials already running to finish, and discards their results;
- under `DistributedQueue`, it removes the workers of the trials still running.

The trials that were still running are marked `Abandoned`, and the interrupt is rethrown. Every completed trial stays in `ho`, so `minimum`, `minimizer`, `DataFrame` and `summaryplot` still work on it. The same happens when something other than the objective throws during a run, e.g. the sampler; an objective that throws only fails its own trial.

A run stopped this way cannot be continued: `run!` and `settarget!` throw on it. To continue, load the last checkpoint, as below. Stopping does not write a checkpoint, so the trials completed since the last one are not in it.

In a script run as `julia script.jl`, Ctrl-C exits Julia straight away instead, unless the script first calls `Base.exit_on_sigint(false)`.

## Saving and resuming

`run!` can checkpoint `ho` as it goes:

```julia
run!(ho; executor=Threaded(), save_every=10, save_path="ho_checkpoint.jld2")
```

With `save_path` set, `ho` is saved there when the run finishes, and also after every `save_every` trials if that is set. `save_hyperoptimizer(ho, path)` saves it at any other point.

A checkpoint stores neither the objective nor the executor, so the objective is given again when loading, and the resumed run can use a different executor:

```julia
ho = load_hyperoptimizer(Stateful(objective), "ho_checkpoint.jld2")
run!(ho; executor=DistributedQueue(8; spawn_worker, setup_worker), save_every=10,
     save_path="ho_checkpoint.jld2")
```

Trials that were still running when the checkpoint was written are marked `Abandoned` on loading, since their results never reached it.

- A checkpoint is written to a temporary file first and then moved into place, so a crash while saving leaves the previous checkpoint intact.
- A trial's error is saved as its printed text.
- A `Continuous` domain's `transform` should be a named function rather than an anonymous one, so the checkpoint can be loaded in a new Julia session.

## Extending a finished run

`settarget!` raises the number of trials of a finished run, and `run!` then continues it:

```julia
ho = Hyperoptimizer(objective, candidates; sampler=SobolSampler(), n=100)
run!(ho)
settarget!(ho, 200)
run!(ho)
```

The target can only grow. `LHSampler` and the successive-halving samplers fix their plan when the `Hyperoptimizer` is built, so they reject `settarget!`; for more trials, build a new `Hyperoptimizer`.

## Progress and results

`run!` shows a progress bar of finished trials; `show_progress=false` turns it off, e.g. when the output goes to a log file.

After a run:

- `minimum(ho)` is the lowest objective value, and `minimizer(ho)` and `printmin(ho)` give the hyperparameters of the trial that reached it;
- `history(ho)` and `results(ho)` are the hyperparameters and values of every completed trial, in the order they completed;
- `DataFrame(ho)` has one row per trial, including failed and abandoned ones, with its status, hyperparameters, value, error, artefacts and metadata;
- printing `ho` shows the search space, a successive-halving sampler's schedule, and the best trial so far.

`summaryplot(ho)` plots the completed trials; see [Plotting](plotting.md).
