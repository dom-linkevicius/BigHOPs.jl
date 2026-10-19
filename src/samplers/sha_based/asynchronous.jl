const SHAsync = SuccessiveHalving{false}

"""
    ASHA(; R, η=3, r_min=1, iterations=1, inner=RandomSampler())

Asynchronous `Hyperband`, with the same schedule and arguments: a trial is promoted as soon as it is among the best `1/η` completed so far in its rung, instead of waiting for the rung to finish (Li et al., 2020, https://doi.org/10.48550/arXiv.1810.05934). This keeps workers busy under `Threaded` or `DistributedQueue`.
"""
const ASHA = SuccessiveHalving{false,<:BasicSamplers}

SuccessiveHalving{false,<:BasicSamplers}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1,
                                         inner::BasicSamplers=RandomSampler()) =
    SuccessiveHalving{false}(; R=R, η=η, r_min=r_min, iterations=iterations, inner=inner)

_n_promotable(s::SHAsync, runs, bracket::ActiveBracket, rung::Int) =
    min(floor(Int, length(_told_sorted(runs, bracket.rungs[rung])) / s.η), bracket.rungs[rung+1].capacity)

function _bracket_decision(s::SHAsync, bracket::ActiveBracket, runs)
    n_rungs = length(bracket.rungs)
    for rung in (n_rungs-1):-1:1
        promoted = _promoted_ids(runs, bracket.rungs[rung+1])
        if length(promoted) < _n_promotable(s, runs, bracket, rung)
            told = _told_sorted(runs, bracket.rungs[rung])
            id, _ = first(t for t in told if first(t) ∉ promoted)
            return _promote(bracket, bracket.rungs[rung], id)
        end
    end
    _dispatched_count(bracket.rungs[1]) < bracket.rungs[1].capacity && return _draw(bracket, bracket.rungs[1])
    any(rung -> _pending_count(runs, rung) > 0, bracket.rungs) && return _wait()
    return _exhausted()
end

function _bracket_has_room(s::SHAsync, bracket::ActiveBracket, runs)
    _dispatched_count(bracket.rungs[1]) < bracket.rungs[1].capacity && return true
    return any(rung -> length(_promoted_ids(runs, bracket.rungs[rung+1])) < _n_promotable(s, runs, bracket, rung),
               1:(length(bracket.rungs)-1))
end

function _rung_resolved(s::SHAsync, runs, bracket::ActiveBracket, rung::Int)
    dispatch_final = if _dispatched_count(bracket.rungs[rung]) == bracket.rungs[rung].capacity
        true
    elseif rung == 1
        false
    else
        _rung_resolved(s, runs, bracket, rung - 1) &&
            _dispatched_count(bracket.rungs[rung]) == _n_promotable(s, runs, bracket, rung - 1)
    end
    return dispatch_final && _pending_count(runs, bracket.rungs[rung]) == 0
end

function on_tell!(s::SHAsync, runs, entry)
    bracket = _bracket_of(s, entry)
    n_rungs = length(bracket.rungs)

    if all(rung -> _pending_count(runs, rung) == 0, bracket.rungs) && !_bracket_has_room(s, bracket, runs)
        total_capacity = sum(rung.capacity for rung in bracket.rungs)
        total_dispatched = sum(_dispatched_count(rung) for rung in bracket.rungs)
        total_dispatched < total_capacity && @warn "$(typeof(s)): $(_label(bracket)) stalled at $total_dispatched/$total_capacity trials dispatched -- no rung can accept more"
    end

    resolved_before = false
    for rung in entry.metadata[:rung]:n_rungs
        if _rung_resolved(s, runs, bracket, rung)
            resolved_before || _rung_has_failure(runs, bracket.rungs[rung]) && @warn "$(typeof(s)): rung $rung of $(_label(bracket)) completed with at least one failed trial"
        end
        rung == n_rungs && break
        resolved_before = _dispatched_count(bracket.rungs[rung+1]) == bracket.rungs[rung+1].capacity ||
                          (resolved_before && _dispatched_count(bracket.rungs[rung+1]) == _n_promotable(s, runs, bracket, rung))
    end
    return nothing
end
