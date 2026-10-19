@enum RunStatus Pending Completed Failed Abandoned

"""
    RunEntry(id, params::NamedTuple, unit_params, metadata=Dict{Symbol,Any}(); pre_artefact=nothing)

One trial: its hyperparameters `params` and their unit-interval encoding `unit_params`, sampler `metadata`, `status` (`Pending`, `Completed`, `Failed` or `Abandoned`), `value`, `error`, and the `pre_artefact` and `post_artefact` of a `Stateful` objective.
"""
struct RunEntry{P<:NamedTuple}
    id::Int
    params::P
    unit_params::Vector{Float64}
    metadata::Dict{Symbol,Any}
    status::RunStatus
    value::Any
    error::Any
    pre_artefact::Any
    post_artefact::Any
end
function RunEntry(id::Int, params::NamedTuple, unit_params::AbstractVector{<:Real},
                  metadata::Dict{Symbol,Any}=Dict{Symbol,Any}(); pre_artefact=nothing)
    return RunEntry(id, params, unit_params, metadata, Pending, missing, nothing, pre_artefact, nothing)
end

_with_result(entry::RunEntry, status::RunStatus, value, post_artefact; error=nothing) =
    RunEntry(entry.id, entry.params, entry.unit_params, entry.metadata, status, value, error, entry.pre_artefact, post_artefact)

"""
    finalize_entry(entry::RunEntry, outcome) -> RunEntry

Return `entry` with `outcome` recorded: a real number completes it, while `NaN`, any other value or an exception marks it `Failed`.
"""
function finalize_entry(entry::RunEntry, outcome::Real)
    if isnan(outcome)
        @warn "Objective returned NaN; excluding this trial as failed" params = entry.params value = outcome
        return _with_result(entry, Failed, missing, entry.post_artefact; error=outcome)
    end
    return _with_result(entry, Completed, outcome, entry.post_artefact)
end
function finalize_entry(entry::RunEntry, outcome)
    @warn "Objective returned a non-Real value; excluding this trial as failed" params = entry.params value = outcome
    return _with_result(entry, Failed, missing, entry.post_artefact; error=outcome)
end

"""
    ObjectiveOutcome(value, post_artefact)

What one objective call produced: its value and, for a `Stateful` objective, the state it returned.
"""
struct ObjectiveOutcome
    value::Any
    post_artefact::Any
end
function finalize_entry(entry::RunEntry, outcome::ObjectiveOutcome)
    told = finalize_entry(entry, outcome.value)
    return _with_result(told, told.status, told.value, outcome.post_artefact; error=told.error)
end

"""
    Stateful(f)

Wrap an objective that carries state between trials: `f(p; pre_artefact)` returns `(value, post_artefact)`. When `Hyperband` or `ASHA` promote a trial to a higher resource level, the promoted trial gets the earlier `post_artefact` as its `pre_artefact`, so it can resume rather than restart; otherwise `pre_artefact` is `nothing`.
"""
struct Stateful{F}
    f::F
end

"""
    call_objective(f, params, pre_artefact)

Call the objective on `params`, passing `pre_artefact` only to a `Stateful` objective, and return an `ObjectiveOutcome`.
"""
call_objective(f, params, pre_artefact) = ObjectiveOutcome(f(params), nothing)
call_objective(s::Stateful, params, pre_artefact) = ObjectiveOutcome(s.f(params; pre_artefact)...)
