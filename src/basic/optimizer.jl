"""
    OptimizerStatus

Lifecycle of a `Hyperoptimizer`: `Initialized`, `Running`, `Finished`, or `Errored`, which cannot be resumed.
"""
@enum OptimizerStatus Initialized Running Errored Finished

mutable struct Hyperoptimizer{S<:Sampler,F}
    params::Vector{Symbol}
    candidates::Tuple
    sampler::S
    objective::F
    n::Int
    runs::Vector{RunEntry}
    completed::Vector{Int}
    n_pending::Int
    status::OptimizerStatus
    best_min_id::Union{Int,Nothing}
    lock::ReentrantLock
end

"""
    Hyperoptimizer(objective, candidates::NamedTuple; sampler=RandomSampler(), n::Int)

Set up a search of `n` trials over `candidates`, a `NamedTuple` of `Domain`s. `objective` takes one `NamedTuple` of hyperparameters and returns the number to minimize; see `Stateful` for objectives that carry state between trials.
"""
function Hyperoptimizer(objective, candidates::NamedTuple; sampler::Sampler=RandomSampler(), n::Int)
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    cands = values(candidates)
    all(d -> d isa Domain, cands) ||
        throw(ArgumentError("every candidate must be a Domain (Continuous/Nominal/Ordinal), got types: $(typeof.(cands))"))
    params = collect(Symbol, keys(candidates))
    initialized_sampler = init(sampler, cands, n)
    return Hyperoptimizer(params, cands, initialized_sampler, objective, n,
                           RunEntry[], Int[], 0, Initialized,
                           nothing, ReentrantLock())
end

"""
    Hyperoptimizer(objective, candidates::NamedTuple, sampler::SuccessiveHalving)

Set up a search with `Hyperband`, `ASHA` or `DEHB`. The number of trials comes from the sampler's schedule, and `objective` receives the resource level as `p.r`, so `:r` cannot be a candidate name.
"""
function Hyperoptimizer(objective, candidates::NamedTuple, sampler::SuccessiveHalving)
    haskey(candidates, :r) &&
        throw(ArgumentError("Hyperoptimizer: `:r` is reserved for $(typeof(sampler))'s resource level -- rename your `:r` candidate"))
    _check_objective(sampler, objective)
    n = _total_trials(sampler)
    return Hyperoptimizer(objective, candidates; sampler=sampler, n=n)
end

reached_target(ho::Hyperoptimizer) = length(ho.runs) >= ho.n

n_told(ho::Hyperoptimizer) = length(ho.runs) - ho.n_pending

"""
    settarget!(ho, n)

Raise the trial target to `n`, e.g. to continue a finished run with `run!`. The target can only grow, and `FixedPlanSampler`s reject it.
"""
function settarget!(ho::Hyperoptimizer, n::Int)
    ho.status == Errored &&
        throw(ArgumentError("settarget!: this Hyperoptimizer already errored and cannot be resumed -- construct a new Hyperoptimizer to continue"))
    ho.n_pending > 0 &&
        @warn "settarget!: $(ho.n_pending) trial(s) still pending -- changing the target while trials are in flight may race with them"
    n < ho.n &&
        throw(ArgumentError("settarget!: new target ($n) is less than the current target ($(ho.n)) -- settarget! can only raise the target"))
    ho.sampler isa FixedPlanSampler &&
        throw(ArgumentError("settarget!: $(typeof(ho.sampler)) fixes its plan to the sample count given at construction and can't respond to a new target"))
    ho.n = n
    @info "Hyperoptimizer target set to $(ho.n) trials"
    return ho
end

"""
    ask!(ho) -> RunEntry

Draw the next trial from the sampler and record it as pending. Throws if the sampler is exhausted, the target is reached, or `ho` has errored.
"""
function ask!(ho::Hyperoptimizer)
    lock(ho.lock) do
        ho.status == Errored &&
            throw(ArgumentError("ask!: this Hyperoptimizer already errored and cannot produce new trials -- construct a new Hyperoptimizer to continue"))
        exhausted(ho.sampler, ho) && throw(ArgumentError("Hyperoptimizer's sampler is exhausted: no more candidates available"))
        reached_target(ho) && throw(ArgumentError("Hyperoptimizer has already reached its target of $(ho.n) trials; call settarget! to raise it before asking for more"))
        unit_params = ho.sampler(ho.candidates, ho.runs)
        id = length(ho.runs) + 1
        decoded = Tuple(from_unit(d, u) for (d, u) in zip(ho.candidates, unit_params))
        params = NamedTuple{Tuple(ho.params)}(decoded)
        entry = create_run_entry(ho.sampler, ho, id, params, unit_params)
        push!(ho.runs, entry)
        ho.n_pending += 1
        return entry
    end
end

function update_best!(ho::Hyperoptimizer, entry::RunEntry)
    if ho.best_min_id === nothing || entry.value < ho.runs[ho.best_min_id].value
        ho.best_min_id = entry.id
    end
    return ho
end

"""
    tell!(ho, entry, outcome)

Record the `outcome` of a trial from `ask!`. A `NaN`, a non-number or an exception marks the trial `Failed` instead of stopping the search.
"""
function tell!(ho::Hyperoptimizer, entry::RunEntry, outcome)
    lock(ho.lock) do
        ho.status == Errored &&
            throw(ArgumentError("tell!: this Hyperoptimizer already errored and cannot record new outcomes -- construct a new Hyperoptimizer to continue"))
        told = finalize_entry(ho.runs[entry.id], outcome)
        ho.runs[told.id] = told
        ho.n_pending -= 1
        if told.status === Completed
            push!(ho.completed, told.id)
            update_best!(ho, told)
        end
        on_tell!(ho.sampler, ho.runs, told)
        return ho
    end
end

_should_stop_asking(ho::Hyperoptimizer) = reached_target(ho) || exhausted(ho.sampler, ho) || blocked(ho.sampler, ho)

"""
    run!(ho; executor=Serial(), save_every=nothing, save_path=nothing, show_progress=true)

Evaluate trials until the target is reached or the sampler is exhausted, using `executor` (`Serial`, `Threaded` or `DistributedQueue`). Given `save_path`, `ho` is checkpointed when the run ends, and also after every `save_every` trials if that is set; resume with `load_hyperoptimizer`.
"""
function run!(ho::Hyperoptimizer; executor::AbstractExecutor=Serial(),
              save_every::Union{Int,Nothing}=nothing, save_path::Union{AbstractString,Nothing}=nothing,
              show_progress::Bool=true)
    save_every !== nothing && save_path === nothing &&
        throw(ArgumentError("run!: save_every requires save_path"))
    save_every !== nothing && save_every < 1 &&
        throw(ArgumentError("run!: save_every must be >= 1, got $save_every"))
    save_path !== nothing && ho.objective === nothing &&
        throw(ArgumentError("run!: save_path requires a real ho.objective -- checkpointing substitutes `nothing` for it in the saved file (to be replaced with a fresh objective via load_hyperoptimizer), which would be ambiguous if it was already `nothing`"))
    ho.status == Errored &&
        throw(ArgumentError("run!: this Hyperoptimizer already errored and cannot be resumed -- construct a new Hyperoptimizer to continue"))
    if reached_target(ho)
        @warn "run!: Hyperoptimizer has already reached its target of $(ho.n) trials; call settarget! to raise it before calling run! again"
        return ho
    end
    if exhausted(ho.sampler, ho)
        if ho.sampler isa FixedPlanSampler
            @warn "$(typeof(ho.sampler)) has a fixed plan and is already exhausted; run! won't produce any more trials -- change params or use a different sampler"
        else
            @warn "run!: $(typeof(ho.sampler)) is exhausted; run! won't produce any more trials"
        end
        return ho
    end
    ho.status = Running
    start!(executor, ho)
    last_saved = n_told(ho)
    progress = nothing
    if show_progress
        progress = ProgressMeter.Progress(ho.n)
        ProgressMeter.update!(progress, n_told(ho); force=true)
    end
    try
        while true
            while !_should_stop_asking(ho) && capacity(executor) > 0
                entry = ask!(ho)
                submit!(executor, entry, ho.objective)
            end
            for (entry, outcome) in poll(executor)
                tell!(ho, entry, outcome)
            end
            if save_every !== nothing && n_told(ho) - last_saved >= save_every
                save_hyperoptimizer(ho, save_path)
                last_saved = n_told(ho)
            end
            progress !== nothing && ProgressMeter.update!(progress, n_told(ho))
            _should_stop_asking(ho) && ho.n_pending == 0 && break
        end
        ho.status = Finished
        save_path !== nothing && save_hyperoptimizer(ho, save_path)
    catch e
        _handle_run_error(e, ho)
    finally
        shutdown!(executor)
        progress !== nothing && ProgressMeter.finish!(progress)
    end
    return ho
end

function _handle_run_error(e, ho::Hyperoptimizer)
    ho.status = Errored
    lock(ho.lock) do
        for (i, entry) in enumerate(ho.runs)
            if entry.status === Pending
                ho.runs[i] = _with_result(entry, Abandoned, missing, entry.post_artefact)
                ho.n_pending -= 1
            end
        end
    end
    rethrow(e)
end
