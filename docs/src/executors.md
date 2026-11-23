# Executors

Executors are the workhorses of `BigHOPs.jl`, they specify how the individual trials are run, launch new ones when necessary. The executor has to be specified when `run!` is called on a `Hyperoptimizer`, with the conscious goal for flexibility and modularity, to be able to run any sampler with any executor:

```julia
ho = Hyperoptimizer(f, domains; sampler=RandomSampler(), n=100)
run!(ho; executor=Threaded())
```

The choice of executor should be informed by the problem at hand. Pick `Serial` for development and debugging, `Threaded` for one machine with multiple cores and intermediate size problems, and `DistributedQueue` for clusters, or trials that need a process of their own or a time limit. Finally, see [General functionalities](@ref) for some additional convenience functionalities related to executors.

Every executor gives each trial its own copy of the objective `f`, of its hyperparameters `x` and of its `pre_artefact`. An objective's internal state therefore does not carry over from one trial to the next, changing `x` in place does not change the stored trial or the domain's values, and changing `pre_artefact` in place does not change the earlier trial's `post_artefact`. Anything the objective captures, e.g. training data, is copied once per trial.

## Serial

```@docs; canonical=false
Serial
```

`Serial` is the default for `run!`. It is the simplest to debug, suits objectives that already use every core of the machine on their own.

## Threaded

```@docs; canonical=false
Threaded
```

The objectives evaluated using `Threaded` have to be safe to run many trials concurrently, for example by not writing to shared global state (`BigHOPs.jl` does not ensure this, therefore it is left for the user).

## DistributedQueue

```@docs; canonical=false
DistributedQueue
```

Each worker is a fresh Julia process, so the objective has to be defined there. A convenient pattern is to keep it in its own file and include that file both locally and in `setup_worker`:

```julia
using Distributed
using BigHOPs

include("objective.jl")  # defines `f(x)`

spawn_worker() = first(addprocs(1; exeflags="--threads=12"))
function setup_worker(pid)
    remotecall_eval(Main, [pid], :(using BigHOPs; include("objective.jl")))
    return nothing
end

ho = Hyperoptimizer(f, domains; sampler=SobolSampler(), n=200)
run!(ho; executor=DistributedQueue(8; spawn_worker, setup_worker, max_single_runtime=3600))
```

The option to spawn and kill processes instead of having one process with a large resource pool was due to various factors, e.g. trials freezing, processes dying and resources being unrecoverable, efficiency of resource usage in cases where initially a sampler can benefit from lots of processes, but only runs few at the end.