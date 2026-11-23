"""
    Threaded(max_concurrency=Threads.nthreads())

Evaluate up to `max_concurrency` trials at once, as tasks spread over Julia's threads. Start Julia with `--threads=N` for real parallelism; on one thread the trials only interleave.
"""
mutable struct Threaded <: AbstractExecutor
    max_concurrency::Int
    results::Channel{Tuple{RunEntry,Any,DateTime,DateTime}}
    in_flight::Int
    pending_exception::Union{Exception,Nothing}
    tasks::Vector{Task}

    function Threaded(max_concurrency::Int, results::Channel{Tuple{RunEntry,Any,DateTime,DateTime}}, in_flight::Int, pending_exception::Union{Exception,Nothing}, tasks::Vector{Task})
        max_concurrency >= 1 || throw(ArgumentError("max_concurrency must be >= 1, got $max_concurrency"))
        return new(max_concurrency, results, in_flight, pending_exception, tasks)
    end
end
function Threaded(max_concurrency::Int=Threads.nthreads())
    Threads.nthreads() == 1 &&
        @warn "Threaded executor constructed with only 1 Julia thread available; trials will run concurrently as tasks but not in true parallel -- start Julia with `--threads=N` for real concurrency" maxlog = 1
    return Threaded(max_concurrency, Channel{Tuple{RunEntry,Any,DateTime,DateTime}}(Inf), 0, nothing, Task[])
end

function start!(executor::Threaded, ho)
    executor.results = Channel{Tuple{RunEntry,Any,DateTime,DateTime}}(Inf)
    executor.in_flight = 0
    executor.pending_exception = nothing
    executor.tasks = Task[]
    return nothing
end

"""
    shutdown!(executor::Threaded)

Wait for every trial this executor started to finish.
"""
function shutdown!(executor::Threaded)
    for t in executor.tasks
        wait(t)
    end
    return nothing
end

function submit!(executor::Threaded, entry::RunEntry, f)
    executor.in_flight += 1
    results = executor.results
    t = Threads.@spawn begin
        start_time = now()
        outcome = try
            safe_call(deepcopy(f), deepcopy(entry.params), deepcopy(entry.pre_artefact))
        catch e
            @warn "objective interrupted" error = e
            e
        end
        put!(results, (entry, outcome, start_time, now()))
    end
    push!(executor.tasks, t)
    return nothing
end

function _take_one!(executor::Threaded, out)
    entry, outcome, start_time, end_time = take!(executor.results)
    executor.in_flight -= 1
    entry.metadata[:start_time] = start_time
    entry.metadata[:end_time] = end_time
    if outcome isa InterruptException
        executor.pending_exception = outcome
    else
        push!(out, (entry, outcome))
    end
    return out
end

function poll(executor::Threaded, timeout::Real=Inf)
    executor.pending_exception !== nothing && throw(executor.pending_exception)
    executor.in_flight == 0 && return Tuple{RunEntry,Any}[]
    isfinite(timeout) && timedwait(() -> isready(executor.results), timeout) === :timed_out && return Tuple{RunEntry,Any}[]
    out = _take_one!(executor, Tuple{RunEntry,Any}[])
    while executor.in_flight > 0 && isready(executor.results)
        _take_one!(executor, out)
    end
    return out
end

interrupt!(executor::Threaded) = empty!(executor.tasks)

capacity(executor::Threaded) = executor.pending_exception === nothing ? executor.max_concurrency - executor.in_flight : 0
