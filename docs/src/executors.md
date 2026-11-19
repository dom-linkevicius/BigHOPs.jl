# Executors

An executor decides how `run!` evaluates trials, while the sampler decides which hyperparameters each trial gets. Any sampler works with any executor, so the executor is chosen when the search is run, not when it is set up:

```julia
ho = Hyperoptimizer(objective, candidates; sampler=RandomSampler(), n=100)
run!(ho; executor=Threaded())
```

## Serial

```julia
Serial()
```

Evaluates one trial at a time, in the calling task. It is the default for `run!`, and the simplest to debug, since everything runs in order in one place. It also suits objectives that already use every core of the machine on their own.

## Threaded

```julia
Threaded(max_concurrency=Threads.nthreads())
```

Evaluates up to `max_concurrency` trials at once, as tasks spread over Julia's threads. Start Julia with `--threads=N` (or `-t N`) for real parallelism; on one thread the trials only interleave.

The trials share one Julia process, so the objective has to be safe to run concurrently with itself, for example by not writing to shared global state.

## DistributedQueue

```julia
DistributedQueue(max_concurrency=Sys.CPU_THREADS; spawn_worker, setup_worker=Returns(nothing),
                 teardown_timeout=30, max_single_runtime=Inf, worker_metadata=Returns(NamedTuple()))
```

Evaluates each trial on its own freshly spawned `Distributed` worker, up to `max_concurrency` at a time, and removes the worker when the trial returns. It is meant for the largest jobs, usually on a compute cluster, where each trial needs a process (or a node) of its own.

- `spawn_worker()` starts one worker and returns its process id, e.g. `() -> first(addprocs(1))`. On a cluster, it can request a process from the scheduler instead.
- `setup_worker(pid)` prepares that worker before its trial, e.g. loads packages and defines the objective there.
- `teardown_timeout` is how many seconds to wait when removing a worker.
- `max_single_runtime` is how many seconds a trial's objective may run before its worker is killed and the trial is marked `TimedOut`. The default `Inf` sets no limit. It also accepts a function of the trial's parameters, e.g. `p -> 30 + 20 * p.r`, to scale the limit with a successive-halving resource level.
- `worker_metadata()` runs on each trial's worker before its objective and returns a `NamedTuple` of extra details to record, e.g. `() -> (job = ENV["SLURM_JOB_ID"],)`. They are stored in the trial's `metadata[:worker]`, alongside the worker id, host, OS process id and start time.

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

## Failures and checkpoints

A failing trial does not stop the search under any executor; see [Trial states](@ref) for how failures are recorded, and [Saving and resuming](@ref) for checkpointing long runs.
