"""
    DistributedQueue(max_concurrency=Sys.CPU_THREADS; spawn_worker, setup_worker=Returns(nothing), teardown_timeout=30)

Evaluate each trial on its own freshly spawned Distributed worker, up to `max_concurrency` at a time, removing the worker when the trial returns. Suited to clusters, where `spawn_worker` can request a process from a scheduler.
"""
mutable struct DistributedQueue <: AbstractExecutor
    max_concurrency::Int
    spawn_worker::Any
    setup_worker::Any
    teardown_timeout::Real
    results::Channel{Tuple{RunEntry,Any}}
    in_flight::Int
    pending_exception::Union{Exception,Nothing}
    tasks::Vector{Tuple{Int,Task}}

    function DistributedQueue(max_concurrency::Int, spawn_worker, setup_worker, teardown_timeout::Real, results::Channel{Tuple{RunEntry,Any}}, in_flight::Int, pending_exception::Union{Exception,Nothing}, tasks::Vector{Tuple{Int,Task}})
        max_concurrency >= 1 || throw(ArgumentError("max_concurrency must be >= 1, got $max_concurrency"))
        teardown_timeout > 0 ||
            throw(ArgumentError("teardown_timeout must be a positive number of seconds (got $teardown_timeout) -- 0 disables Distributed.rmprocs's blocking wait entirely, silently reinstating an unbounded worker_lock hold"))
        return new(max_concurrency, spawn_worker, setup_worker, teardown_timeout, results, in_flight, pending_exception, tasks)
    end
end
"""
    DistributedQueue(max_concurrency=Sys.CPU_THREADS; spawn_worker, setup_worker=Returns(nothing), teardown_timeout=30)

- `spawn_worker()`: start one worker and return its process id, e.g. `() -> first(addprocs(1))`.
- `setup_worker(pid)`: prepare that worker before its trial, e.g. load packages and define the objective there.
- `teardown_timeout`: seconds to wait when removing a worker.
"""
function DistributedQueue(max_concurrency::Int=Sys.CPU_THREADS; spawn_worker, setup_worker=Returns(nothing), teardown_timeout::Real=30)
    return DistributedQueue(max_concurrency, spawn_worker, setup_worker, teardown_timeout, Channel{Tuple{RunEntry,Any}}(Inf), 0, nothing, Tuple{Int,Task}[])
end

function start!(executor::DistributedQueue, ho)
    executor.results = Channel{Tuple{RunEntry,Any}}(Inf)
    executor.in_flight = 0
    executor.pending_exception = nothing
    executor.tasks = Tuple{Int,Task}[]
    return nothing
end

"""
    shutdown!(executor::DistributedQueue)

Remove the workers of trials still running, waiting up to `teardown_timeout` seconds for each, then wait for their tasks to finish.
"""
function shutdown!(executor::DistributedQueue)
    err = nothing
    for (pid, t) in executor.tasks
        if !istaskdone(t)
            try
                Distributed.rmprocs(pid; waitfor=executor.teardown_timeout)
            catch e
                err === nothing && (err = e)
            end
        end
    end
    for (_, t) in executor.tasks
        wait(t)
    end
    err === nothing || throw(err)
    return nothing
end

function submit!(executor::DistributedQueue, entry::RunEntry, f)
    executor.in_flight += 1
    worker = try
        executor.spawn_worker()
    catch e
        @warn "spawn_worker() failed" error = e
        e
    end
    if worker isa Exception
        put!(executor.results, (entry, worker))
        return nothing
    end
    t = @async begin
        outcome = try
            executor.setup_worker(worker)
            future = Distributed.remotecall(safe_call, worker, f, entry.params, entry.pre_artefact)
            fetch(future)
        catch e
            @warn "setup_worker() or objective errored" error = e
            e
        end
        put!(executor.results, (entry, outcome))
        Distributed.rmprocs(worker; waitfor=executor.teardown_timeout)
    end
    push!(executor.tasks, (worker, t))
    return nothing
end

function _take_one!(executor::DistributedQueue, out)
    entry, outcome = take!(executor.results)
    executor.in_flight -= 1
    if outcome isa InterruptException
        executor.pending_exception = outcome
    else
        push!(out, (entry, outcome))
    end
    return out
end

function poll(executor::DistributedQueue)
    executor.pending_exception !== nothing && throw(executor.pending_exception)
    executor.in_flight == 0 && return Tuple{RunEntry,Any}[]
    out = _take_one!(executor, Tuple{RunEntry,Any}[])
    while executor.in_flight > 0 && isready(executor.results)
        _take_one!(executor, out)
    end
    return out
end

capacity(executor::DistributedQueue) = executor.pending_exception === nothing ? executor.max_concurrency - executor.in_flight : 0
