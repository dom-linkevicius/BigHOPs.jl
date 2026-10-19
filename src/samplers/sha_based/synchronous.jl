const SHSync = SuccessiveHalving{true}

"""
    Hyperband(; R, η=3, r_min=1, iterations=1, inner=RandomSampler())

Successive halving over brackets that start at resource levels from `r_min` up to `R`: after each rung the best `1/η` of trials continue with `η` times the resource, and a rung must finish before any of its trials are promoted. `iterations` repeats the whole schedule, `inner` draws the starting configurations, and the objective receives the resource level as `p.r`.
"""
const Hyperband = SuccessiveHalving{true,<:BasicSamplers}

SuccessiveHalving{true,<:BasicSamplers}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1,
                                        inner::BasicSamplers=RandomSampler()) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, iterations=iterations, inner=inner)

function _bracket_decision(s::SHSync, bracket::ActiveBracket, runs)
    n_rungs = length(bracket.rungs)
    _dispatched_count(bracket.rungs[1]) < bracket.rungs[1].capacity && return _draw(bracket, bracket.rungs[1])
    for rung in 1:(n_rungs-1)
        _rung_resolved(s, runs, bracket, rung) || return _wait()
        told = _told_sorted(runs, bracket.rungs[rung])
        n_to_promote = min(bracket.rungs[rung+1].capacity, length(told))
        n_to_promote == 0 && return _exhausted()
        n_promoted = _dispatched_count(bracket.rungs[rung+1])
        n_promoted < n_to_promote && return _promote(bracket, bracket.rungs[rung], first(told[n_promoted+1]))
    end
    _rung_resolved(s, runs, bracket, n_rungs) || return _wait()
    return _exhausted()
end

function _rung_resolved(s::SHSync, runs, bracket::ActiveBracket, rung::Int)
    if rung == 1
        n_expected = bracket.rungs[1].capacity
    else
        _rung_resolved(s, runs, bracket, rung - 1) || return false
        n_expected = min(bracket.rungs[rung].capacity, length(_told_sorted(runs, bracket.rungs[rung-1])))
    end
    return _dispatched_count(bracket.rungs[rung]) == n_expected && _pending_count(runs, bracket.rungs[rung]) == 0
end

function on_tell!(s::SHSync, runs, entry)
    bracket = _bracket_of(s, entry)
    rung = entry.metadata[:rung]
    rung < length(bracket.rungs) || return nothing
    _rung_resolved(s, runs, bracket, rung) || return nothing
    told = _told_sorted(runs, bracket.rungs[rung])
    if isempty(told)
        @warn "$(typeof(s)): every trial at rung $rung of $(_label(bracket)) failed -- abandoning it"
    else
        wanted = bracket.rungs[rung+1].capacity
        length(told) < wanted && @warn "$(typeof(s)): only $(length(told))/$wanted trials completed at rung $rung of $(_label(bracket)) -- promoting fewer than planned into rung $(rung + 1)"
    end
    return nothing
end
