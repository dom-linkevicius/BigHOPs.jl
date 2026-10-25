struct ActiveRung
    rung::Int
    capacity::Int
    resource::Int
    ids::Vector{Int}
end

struct ActiveBracket
    iteration::Int
    bracket::Int
    rungs::Vector{ActiveRung}
end

struct SuccessiveHalving{Sync,S<:Sampler} <: Sampler
    R::Int
    r_min::Int
    η::Int
    iterations::Int
    inner::S
    active_brackets::Vector{ActiveBracket}
    current_itr::Ref{Int}
    last_bracket::Ref{Int}
end

function SuccessiveHalving{Sync}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1,
                                 inner::Sampler=RandomSampler()) where {Sync}
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    η > 1 || throw(ArgumentError("η must be greater than 1, got $η"))
    r_min > 0 || throw(ArgumentError("r_min must be positive, got $r_min"))
    r_min <= R || throw(ArgumentError("r_min must be <= R, got r_min=$r_min, R=$R"))
    iterations > 0 || throw(ArgumentError("iterations must be positive, got $iterations"))
    r_top = r_min * η^_smax(R, r_min, η)
    r_top == R ||
        @warn "SuccessiveHalving: the top resource level reached is $r_top, short of the requested R=$R -- smax=⌊log_η(R/r_min)⌋ floors to the nearest integer, so the schedule only lands exactly on R when R/r_min is an exact power of η"
    return SuccessiveHalving{Sync,typeof(inner)}(R, r_min, η, iterations, inner, ActiveBracket[], Ref(1), Ref(0))
end

_add_r(params::NamedTuple, r::Int) = merge((r=r,), params)

_smax(R::Int, r_min::Int, η::Int) = ndigits(R ÷ r_min; base=η) - 1
_n_rungs(R::Int, r_min::Int, η::Int, bracket::Int) = _smax(R, r_min, η) + 2 - bracket
function _capacity(R::Int, r_min::Int, η::Int, bracket::Int, rung::Int)
    smax = _smax(R, r_min, η)
    n0 = ceil(Int, (smax + 1) * η^(smax + 1 - bracket) / _n_rungs(R, r_min, η, bracket))
    return max(1, floor(Int, n0 / η^(rung - 1)))
end
_resource(::Int, r_min::Int, η::Int, bracket::Int, rung::Int) = r_min * η^(bracket + rung - 2)

function _total_trials(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, bracket, rung)
               for bracket in 1:(smax+1) for rung in 1:_n_rungs(R, r_min, η, bracket))
end
function _total_draws(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, bracket, 1) for bracket in 1:(smax+1))
end
_total_trials(s::SuccessiveHalving) = s.iterations * _total_trials(s.R, s.r_min, s.η)
_total_draws(s::SuccessiveHalving) = s.iterations * _total_draws(s.R, s.r_min, s.η)

_dispatched_count(rung::ActiveRung) = length(rung.ids)
_pending_count(runs, rung::ActiveRung) = count(id -> runs[id].status === Pending, rung.ids)
_rung_has_failure(runs, rung::ActiveRung) = any(id -> runs[id].status in (Failed, TimedOut), rung.ids)
_told_sorted(runs, rung::ActiveRung) =
    sort([(id, runs[id].value) for id in rung.ids if runs[id].status === Completed]; by=last)
_promoted_ids(runs, rung::ActiveRung) = Set(runs[id].metadata[:promoted_from] for id in rung.ids)

function _make_bracket(s::SuccessiveHalving, iteration::Int, bracket::Int)
    rungs = [ActiveRung(rung, _capacity(s.R, s.r_min, s.η, bracket, rung),
                        _resource(s.R, s.r_min, s.η, bracket, rung), Int[])
             for rung in 1:_n_rungs(s.R, s.r_min, s.η, bracket)]
    return ActiveBracket(iteration, bracket, rungs)
end

function _open_next!(s::SuccessiveHalving)
    if s.last_bracket[] < _smax(s.R, s.r_min, s.η) + 1
        s.last_bracket[] += 1
    elseif s.current_itr[] < s.iterations
        s.current_itr[] += 1
        s.last_bracket[] = 1
    else
        return nothing
    end
    push!(s.active_brackets, _make_bracket(s, s.current_itr[], s.last_bracket[]))
    return last(s.active_brackets)
end

struct SHDecision{K}
    bracket::Union{ActiveBracket,Missing}
    rung::Union{ActiveRung,Missing}
    promoted_from::Union{Int,Missing}
end

_draw(bracket::ActiveBracket, rung::ActiveRung) = SHDecision{:draw}(bracket, rung, missing)
_promote(bracket::ActiveBracket, rung::ActiveRung, promoted_from::Int) =
    SHDecision{:promote}(bracket, rung, promoted_from)
_wait() = SHDecision{:wait}(missing, missing, missing)
_exhausted() = SHDecision{:exhausted}(missing, missing, missing)

function _manage_decide!(s::SuccessiveHalving, runs)
    while true
        isempty(s.active_brackets) && _open_next!(s) === nothing && return _exhausted()
        decision = _bracket_decision(s, first(s.active_brackets), runs)
        decision isa SHDecision{:exhausted} || return decision
        popfirst!(s.active_brackets)
    end
end

_propose(d::SHDecision{:draw}, s::SuccessiveHalving, candidates, runs) = _sample_sh_inner(s, candidates, runs, d)
_propose(d::SHDecision{:promote}, ::SuccessiveHalving, candidates, runs) = copy(runs[d.promoted_from].unit_params)
_propose(::SHDecision{:wait}, s::SuccessiveHalving, candidates, runs) =
    throw(ArgumentError("$(typeof(s)) has nothing to propose right now -- every bracket is waiting on trials that were asked but not yet told; `blocked` reports this"))
_propose(::SHDecision{:exhausted}, s::SuccessiveHalving, candidates, runs) =
    throw(ArgumentError("$(typeof(s)) has finished all $(s.iterations) iterations of its schedule and can propose nothing further; `exhausted` reports this"))

_metadata(bracket::ActiveBracket, rung::Int) =
    Dict{Symbol,Any}(:iteration => bracket.iteration, :bracket => bracket.bracket, :rung => rung)

function _entry_for(d::SHDecision{:draw}, s::SuccessiveHalving, ho, id, params, unit_params)
    return RunEntry(id, _add_r(params, d.rung.resource), unit_params, _metadata(d.bracket, d.rung.rung))
end

function _entry_for(d::SHDecision{:promote}, s::SuccessiveHalving, ho, id, params, unit_params)
    target = d.bracket.rungs[d.rung.rung+1]
    metadata = _metadata(d.bracket, target.rung)
    metadata[:promoted_from] = d.promoted_from
    return RunEntry(id, _add_r(params, target.resource), unit_params, metadata;
                    pre_artefact=ho.runs[d.promoted_from].post_artefact)
end

function _bracket_of(s::SuccessiveHalving, entry)
    idx = findfirst(s.active_brackets) do bracket
        bracket.iteration == entry.metadata[:iteration] && bracket.bracket == entry.metadata[:bracket]
    end
    idx === nothing &&
        throw(ArgumentError("$(typeof(s)): trial $(entry.id) reports bracket $(entry.metadata[:bracket]) of iteration $(entry.metadata[:iteration]), which is no longer active; a bracket is only closed once none of its rungs has a pending trial"))
    return s.active_brackets[idx]
end

_label(bracket::ActiveBracket) = "bracket $(bracket.bracket) of iteration $(bracket.iteration)"

function (s::SuccessiveHalving)(candidates, runs)
    return _propose(_manage_decide!(s, runs), s, candidates, runs)
end

_sample_sh_inner(s::SuccessiveHalving, candidates, runs, ::SHDecision{:draw}) = _sample_sh_inner(s.inner, candidates, runs)
_sample_sh_inner(inner::Sampler, candidates, runs) = inner(candidates, runs)
_sample_sh_inner(inner::LHSampler, candidates, runs) = inner(candidates, filter(e -> e.metadata[:rung] == 1, runs))

function init(s::SuccessiveHalving{Sync}, candidates, n) where {Sync}
    inner = init(s.inner, candidates, _total_draws(s))
    return SuccessiveHalving{Sync,typeof(inner)}(s.R, s.r_min, s.η, s.iterations, inner,
                                                 ActiveBracket[], Ref(1), Ref(0))
end
exhausted(s::SuccessiveHalving, ho) = _manage_decide!(s, ho.runs) isa SHDecision{:exhausted}
blocked(s::SuccessiveHalving, ho) = _manage_decide!(s, ho.runs) isa SHDecision{:wait}

function _record!(bracket::ActiveBracket, rung::ActiveRung, id::Int)
    length(rung.ids) < rung.capacity ||
        throw(ArgumentError("$(_label(bracket)) rung $(rung.rung) already holds its capacity of $(rung.capacity) trials; dispatching another would break the schedule"))
    push!(rung.ids, id)
    return nothing
end
_record!(d::SHDecision{:draw}, id::Int) = _record!(d.bracket, d.rung, id)
_record!(d::SHDecision{:promote}, id::Int) = _record!(d.bracket, d.bracket.rungs[d.rung.rung+1], id)

_check_objective(s::SuccessiveHalving, objective) = objective isa Stateful || objective === nothing || @warn "$(typeof(s)) with a non-Stateful objective: promoted trials can't " *
                                                             "resume from a previous trial's state, so each promotion re-pays all the resource already spent on it -- " *
                                                             "wrap the objective in `Stateful` to make promotions continue instead of restart"

function create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    decision = _manage_decide!(s, ho.runs)
    entry = _entry_for(decision, s, ho, id, params, unit_params)
    _record!(decision, id)
    return entry
end
