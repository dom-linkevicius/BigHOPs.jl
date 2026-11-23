# General functionalities

In addition to domains, samplers and executors, `BigHOPs.jl` has features that make long or expensive searches easier to run, resume, diagnose and analyze.

## Trial states

Every trial (attempted evaluation of `f(x)`) can have one of five states:

- `Pending`: the trial has been handed to the executor and has not finished yet.
- `Completed`: the objective returned a real number, which is the trial's `value`.
- `Failed`: the objective threw an exception, or returned `NaN` or something other than a number. Under `DistributedQueue`, a trial also fails when `spawn_worker` or `setup_worker` throws for it.
- `TimedOut`: under `DistributedQueue`, the trial ran longer than `max_single_runtime`, and its worker was killed.
- `Abandoned`: the trial was still running when its run stopped early, or when the checkpoint it was loaded from was written, so its result was never recorded.

A failing trial does not stop the search under any executor; it is recorded, and the run continues. Its hyperparameters stay in the results too, so it can be inspected for debugging and reproduced.

Only completed trials have a value, so `minimum`, `minimizer`, `history` and `results` use only them, but every trial counts towards the target number of objective evaluations, whatever its state.

## Progress and results

`run!` shows a progress bar of finished trials; `show_progress=false` turns it off, e.g. when the output goes to a log file.

After a run:

- `minimum(ho)` is the lowest objective value, and `minimizer(ho)` and `printmin(ho)` give the hyperparameters of the trial that reached it;
- `history(ho)` and `results(ho)` are the hyperparameters and values of every completed trial, in the order they completed;
- printing `ho` shows the search space, a successive-halving sampler's schedule, and the best trial so far.

## Graceful termination

A run can be stopped at any time with Ctrl-C. `run!` then stops starting new trials and shuts the executor down:

- under `Serial`, the trial that was interrupted is the only one running;
- under `Threaded`, it waits for the trials already running to finish, and discards their results;
- under `DistributedQueue`, it removes the workers of the trials still running.

The trials that were still running are marked `Abandoned`, and the interrupt is rethrown. Every completed trial stays in `ho`, so `minimum`, `minimizer`, `DataFrame` and `summaryplot` still work on it. The same happens when something other than the objective throws during a run, e.g. the sampler; an objective that throws only fails its own trial. Also, Ctrl-C doesn't write a checkpoint, so the trials completed since the last one are lost if you resume from it. 

A run stopped this way cannot be continued: `run!` and `settarget!` throw on it. To continue a stopped run you would need to use checkpointing (see below).

If a script is run as `julia script.jl`, Ctrl-C exits Julia at once, with no graceful termination, unless the script first calls `Base.exit_on_sigint(false)`.

## Saving and resuming

`run!` can checkpoint `ho` as it goes:

```julia
run!(ho; executor=Threaded(), save_every=10, save_path="ho_checkpoint.jld2")
```

With `save_path` set, `ho` is saved there when the run finishes, and also after every `save_every` trials if that is set. `save_hyperoptimizer(ho, path)` saves it at any other point, e.g. after a finished run where no checkpointing was set.

A checkpoint stores neither the objective (due to brittleness from closures) nor the executor, so the objective has to be given again when loading, and the resumed run can use a different executor:

```julia
ho = load_hyperoptimizer(Stateful(f), "ho_checkpoint.jld2")
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
ho = Hyperoptimizer(f, domains; sampler=SobolSampler(), n=100)
run!(ho)
settarget!(ho, 200)
run!(ho)
```

`LHSampler` and the successive-halving samplers fix their plan when the `Hyperoptimizer` is built, so they reject `settarget!`; for more trials, build a new `Hyperoptimizer`.

## Stateful objectives

A plain objective takes one `NamedTuple` of hyperparameters and returns the number to minimize. Wrapping it in `Stateful` lets it also receive and return state:

```julia
function f(x; pre_artefact=nothing)
    model, epochs_done = pre_artefact === nothing ? (new_model(x), 0) : pre_artefact
    train!(model, x.r - epochs_done)
    return validation_loss(model), (model, x.r)
end

ho = Hyperoptimizer(Stateful(f), domains, Hyperband(R=81, η=3, r_min=1))
```

A `Stateful` objective is called as `f(x; pre_artefact)` and returns a tuple `(value, post_artefact)`. `value` is minimized as usual, and both artefacts can be anything, usually a `NamedTuple` or some custom `struct`.

### pre_artefact

`pre_artefact` is the state a trial starts from. It is `nothing`, except when a successive-halving sampler promotes a trial to a higher resource level. `Hyperband`, `ASHA` and `BOHB` pass the earlier trial's `post_artefact` as the promoted trial's `pre_artefact`, so training can continue where it stopped instead of paying again for the resource already spent. The objective has to handle both cases, as in the example above.

`Hyperband`, `ASHA` and `BOHB` warn when the objective is not `Stateful`, since every promotion then starts over. `DEHB` trains every trial from scratch and never sets `pre_artefact`, so it warns when the objective is `Stateful`.

Under `Serial` and `Threaded`, `pre_artefact` is the very object the earlier trial returned, so changing it in place, as `train!` does above, also changes what is stored for the earlier trial. Copy it first if the earlier state should be kept. Under `DistributedQueue`, each worker receives its own copy.

### post_artefact

Whatever a `Stateful` objective returns as `post_artefact` is stored with its trial, whichever sampler is used. With a full-resource sampler, this is a way to keep something from every trial, such as the trained model, its training curve or metrics of interest, without having to save it from inside the objective. The artefacts can be easily accessed by converting a `Hyperoptimizer` to a `DataFrame` (see below). Every `post_artefact` is kept in memory for the whole run and written to every checkpoint, so it should be small enough to keep one per trial, e.g. a model's parameters rather than a copy of its training data.

## DataFrame of trials

After a hyperoptimization run is completed casting the `Hyperoptimizer` into a `DataFrame` is a convenient way to conduct downstream analysis. It can be done by calling `DataFrame(ho)` which has one row per trial, including failed and abandoned ones, with its status, hyperparameters, value, error, artefacts and metadata. The `status` column holds the trial's state, and for a `Failed` or `TimedOut` trial, the `error` column holds what went wrong: the exception, or whatever the objective returned instead of a number.

The states are not exported, so filtering by them needs the module name:

```julia
using DataFrames

df = DataFrame(ho)
failed = filter(:status => ==(BigHOPs.Failed), df)
```
