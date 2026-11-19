# Executors

Here we only describe the basic ideas behind the executors. For more details, including functionalities that apply to all of them, see [General functionalities](@ref).

An executor decides how `run!` evaluates trials, while the sampler decides which hyperparameters each trial gets. Any sampler works with any executor, so the executor is chosen when the search is run:

```julia
ho = Hyperoptimizer(objective, candidates; sampler=RandomSampler(), n=100)
run!(ho; executor=Threaded())
```

## Serial

```@docs; canonical=false
Serial
```

It is the default for `run!`, and the simplest to debug. It also suits objectives that already use every core of the machine on their own.

## Threaded

```@docs; canonical=false
Threaded
```

The trials share one Julia process, so the objective has to be safe to run concurrently with itself, for example by not writing to shared global state.

## DistributedQueue

```@docs; canonical=false
DistributedQueue
```

Each worker is a fresh Julia process, so the objective has to be defined there too. A convenient pattern is to keep it in its own file and include that file both locally and in `setup_worker`:

```julia
using Distributed
using BigHOPs

include("objective.jl")  # defines `objective(p)`

spawn_worker() = first(addprocs(1; exeflags="--threads=12"))
function setup_worker(pid)
    remotecall_eval(Main, [pid], :(using BigHOPs; include("objective.jl")))
    return nothing
end

ho = Hyperoptimizer(objective, candidates; sampler=SobolSampler(), n=200)
run!(ho; executor=DistributedQueue(8; spawn_worker, setup_worker, max_single_runtime=3600))
```
