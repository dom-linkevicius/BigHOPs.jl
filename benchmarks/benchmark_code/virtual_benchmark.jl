using BigHOPs
using Dates: DateTime, Millisecond, now

mutable struct VirtualBenchmarkExecutor <: BigHOPs.AbstractExecutor
    max_concurrency::Int
    origin::DateTime
    clock::Float64
    last_exit::Float64
    submitted::Int
    in_flight::Vector{Tuple{Float64,Int,BigHOPs.RunEntry,Any}}

    function VirtualBenchmarkExecutor(max_concurrency::Int, origin::DateTime, clock::Float64, last_exit::Float64, submitted::Int, in_flight::Vector{Tuple{Float64,Int,BigHOPs.RunEntry,Any}})
        max_concurrency >= 1 || throw(ArgumentError("max_concurrency must be >= 1, got $max_concurrency"))
        return new(max_concurrency, origin, clock, last_exit, submitted, in_flight)
    end
end
VirtualBenchmarkExecutor(max_concurrency::Int) =
    VirtualBenchmarkExecutor(max_concurrency, now(), 0.0, time(), 0, Tuple{Float64,Int,BigHOPs.RunEntry,Any}[])

function BigHOPs.start!(executor::VirtualBenchmarkExecutor, ho)
    executor.origin = now()
    executor.clock = 0.0
    executor.submitted = 0
    executor.in_flight = Tuple{Float64,Int,BigHOPs.RunEntry,Any}[]
    executor.last_exit = time()
    return nothing
end

BigHOPs.shutdown!(::VirtualBenchmarkExecutor; interrupt::Bool=false) = nothing

_training_time(::Exception) = 0.0
function _training_time(outcome::BigHOPs.ObjectiveOutcome)
    artefact = outcome.post_artefact
    hasproperty(artefact, :training_time) ||
        throw(ArgumentError("VirtualBenchmarkExecutor needs a Stateful objective whose post_artefact has a training_time field, got a post_artefact of type $(typeof(artefact))"))
    t = artefact.training_time
    t isa Real && isfinite(t) && t >= 0 ||
        throw(ArgumentError("training_time must be a finite, non-negative number of seconds, got $t"))
    return Float64(t)
end

_virtual_datetime(executor::VirtualBenchmarkExecutor, t::Float64) = executor.origin + Millisecond(round(Int, 1000 * t))

function BigHOPs.submit!(executor::VirtualBenchmarkExecutor, entry::BigHOPs.RunEntry, f)
    executor.clock += time() - executor.last_exit
    outcome = BigHOPs.safe_call(deepcopy(f), deepcopy(entry.params), deepcopy(entry.pre_artefact))
    finish = executor.clock + _training_time(outcome)
    entry.metadata[:start_time] = _virtual_datetime(executor, executor.clock)
    entry.metadata[:end_time] = _virtual_datetime(executor, finish)
    executor.submitted += 1
    push!(executor.in_flight, (finish, executor.submitted, entry, outcome))
    executor.last_exit = time()
    return nothing
end

function BigHOPs.poll(executor::VirtualBenchmarkExecutor, timeout::Real=Inf)
    executor.clock += time() - executor.last_exit
    sort!(executor.in_flight; by=t -> (t[1], t[2]))
    if !isempty(executor.in_flight)
        executor.clock = max(executor.clock, first(first(executor.in_flight)))
    end
    finished = splice!(executor.in_flight, 1:count(t -> t[1] <= executor.clock, executor.in_flight))
    executor.last_exit = time()
    return Tuple{BigHOPs.RunEntry,Any}[(entry, outcome) for (_, _, entry, outcome) in finished]
end

BigHOPs.capacity(executor::VirtualBenchmarkExecutor) = executor.max_concurrency - length(executor.in_flight)
