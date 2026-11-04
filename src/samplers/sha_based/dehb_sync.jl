"""
    DEHBSampler(; F=0.5, crossover=0.5, rng=StableRNG(1))

The differential-evolution sampler inside `DEHB`; construct `DEHB` rather than using this directly. `F` scales mutations, and `crossover` is the probability of taking each hyperparameter from the mutant.
"""
struct DEHBSampler{T<:Random.AbstractRNG} <: Sampler
    F::Float64
    crossover::Float64
    rng::T
    subpops::Dict{Int,Vector{Int}}
    total_dispatched::Dict{Int,Int}
    promotion_queues::Dict{Int,Vector{Int}}
end
function DEHBSampler(; F::Real=0.5, crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1))
    0 < F <= 1 || throw(ArgumentError("F must be in (0, 1], got $F"))
    0 <= crossover <= 1 || throw(ArgumentError("crossover must be in [0,1], got $crossover"))
    return DEHBSampler(Float64(F), Float64(crossover), rng, Dict{Int,Vector{Int}}(), Dict{Int,Int}(), Dict{Int,Vector{Int}}())
end

init(s::DEHBSampler, candidates, n) = s

_dehb_standalone() = throw(ArgumentError("DEHBSampler can't be used on its own; construct DEHB(; R, ...) instead"))
on_tell!(::DEHBSampler, runs, entry) = _dehb_standalone()
exhausted(::DEHBSampler, ho) = _dehb_standalone()
blocked(::DEHBSampler, ho) = _dehb_standalone()
create_run_entry(::DEHBSampler, ho, id, params, unit_params) = _dehb_standalone()
(::DEHBSampler)(candidates, runs) = _dehb_standalone()

"""
    DEHB(; R, η=3, r_min=1, iterations=1, F=0.5, crossover=0.5, rng=StableRNG(1))

`Hyperband`'s schedule with differential evolution in place of random draws (Awad et al., 2021, https://doi.org/10.24963/ijcai.2021/296), following the reference implementation (https://github.com/automl/DEHB) except that the first rung of the first bracket is drawn uniformly at random, as in the paper. Each resource level keeps a subpopulation, and every other first-rung trial is evolved from its own level's subpopulation; a trial whose subpopulation slot is still empty is crossed with a random target instead. In the first iteration, later rungs promote the best configurations from the level below unchanged, so a `Stateful` objective receives their `pre_artefact`; in later iterations every trial is evolved from the level below and trains from scratch.
"""
const DEHB = SuccessiveHalving{true,<:DEHBSampler}

SuccessiveHalving{true,<:DEHBSampler}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1, F::Real=0.5,
                                       crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1)) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, iterations=iterations,
                            inner=DEHBSampler(; F=F, crossover=crossover, rng=rng))

_check_objective(s::DEHB, objective) = objective isa Stateful && @warn "$(typeof(s)) with a Stateful objective: pre_artefact is only set on promotions in the first iteration; every other trial trains from scratch. post_artefact is recorded either way"

function _subpop_size(s::DEHB, r::Int)
    smax = _smax(s.R, s.r_min, s.η)
    return maximum(_capacity(s.R, s.r_min, s.η, bracket, rung)
                   for bracket in 1:(smax+1) for rung in 1:_n_rungs(s.R, s.r_min, s.η, bracket)
                   if _resource(s.R, s.r_min, s.η, bracket, rung) == r)
end

function _subpopulations!(s::DEHB)
    de = s.inner
    isempty(de.total_dispatched) || return de
    for bracket in 1:(_smax(s.R, s.r_min, s.η)+1)
        r = _resource(s.R, s.r_min, s.η, bracket, 1)
        de.subpops[r] = zeros(Int, _subpop_size(s, r))
        de.total_dispatched[r] = 0
    end
    return de
end

_next_slot(s::DEHB, r::Int) = mod(s.inner.total_dispatched[r], length(s.inner.subpops[r])) + 1

_first_bracket(bracket::ActiveBracket) = bracket.iteration == 1 && bracket.bracket == 1
_promotion_phase(bracket::ActiveBracket) = bracket.iteration == 1

function _mutant(de::DEHBSampler, target::Vector{Float64}, pool::Vector{Vector{Float64}})
    pool = copy(pool)
    i = findfirst(==(target), pool)
    if i !== nothing && length(pool) > 1
        deleteat!(pool, i)
    end
    append!(pool, [rand(de.rng, length(target)) for _ in 1:(3-length(pool))])
    r1, r2, r3 = pool[randperm(de.rng, length(pool))[1:3]]
    return r1 .+ de.F .* (r2 .- r3)
end

function _crossover(de::DEHBSampler, target::Vector{Float64}, mutant::Vector{Float64})
    cross = rand(de.rng, length(target)) .< de.crossover
    if !any(cross)
        cross[rand(de.rng, eachindex(target))] = true
    end
    return ifelse.(cross, mutant, target)
end

function _de_trial(de::DEHBSampler, target::Vector{Float64}, pool::Vector{Vector{Float64}})
    trial = _crossover(de, target, _mutant(de, target, pool))
    out_of_range = .!(0 .<= trial .<= 1)
    trial[out_of_range] .= rand(de.rng, count(out_of_range))
    return trial
end

_ranked(runs, ids) = ids[sortperm([runs[id].value for id in ids])]

function _mutation_pool(s::DEHB, runs, r::Int, n::Int, d::Int)
    ranked = _ranked(runs, filter(!=(0), s.inner.subpops[r]))
    pool = Vector{Float64}[runs[id].unit_params for id in ranked[1:min(n, end)]]
    return append!(pool, [rand(s.inner.rng, d) for _ in 1:(n-length(pool))])
end

function _global_population(s::DEHB, runs, d::Int)
    de = s.inner
    return Vector{Float64}[id == 0 ? rand(de.rng, d) : runs[id].unit_params for level in sort(collect(keys(de.subpops))) for id in de.subpops[level]]
end

function _evolve(s::DEHB, runs, d::Int, r::Int, r_parents::Int, n::Int)
    de = _subpopulations!(s)
    occupant = de.subpops[r][_next_slot(s, r)]
    target = occupant == 0 ? rand(de.rng, d) : runs[occupant].unit_params
    pool = _mutation_pool(s, runs, r_parents, n, d)
    if length(pool) < 3
        everyone = _global_population(s, runs, d)
        append!(pool, [_mutant(de, target, everyone) for _ in 1:(4-length(pool))])
    end
    return _de_trial(de, target, pool)
end

function _promotion_candidate!(s::DEHB, runs, decision::SHDecision{:promote})
    de = _subpopulations!(s)
    lower, higher = decision.rung, decision.bracket.rungs[decision.rung.rung+1]
    ranked = _ranked(runs, filter(!=(0), de.subpops[lower.resource]))
    queue = get!(de.promotion_queues, higher.resource, Int[])
    if isempty(queue)
        present = [runs[id].unit_params for id in de.subpops[higher.resource] if id != 0]
        append!(queue, [id for id in ranked if runs[id].unit_params ∉ present])
        resize!(queue, min(length(queue), higher.capacity))
    end
    return isempty(queue) ? first(ranked) : first(queue)
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:draw})
    _subpopulations!(s)
    _first_bracket(d.bracket) && return rand(s.inner.rng, length(candidates))
    return _evolve(s, runs, length(candidates), d.rung.resource, d.rung.resource, d.rung.capacity)
end

function _propose(d::SHDecision{:promote}, s::DEHB, candidates, runs)
    _promotion_phase(d.bracket) && return copy(runs[_promotion_candidate!(s, runs, d)].unit_params)
    higher = d.bracket.rungs[d.rung.rung+1]
    return _evolve(s, runs, length(candidates), higher.resource, d.rung.resource, higher.capacity)
end

function _entry_for(d::SHDecision{:promote}, s::DEHB, ho, id, params, unit_params)
    target = d.bracket.rungs[d.rung.rung+1]
    metadata = _metadata(d.bracket, target.rung)
    _promotion_phase(d.bracket) || return RunEntry(id, _add_r(params, target.resource), unit_params, metadata)
    promoted_from = _promotion_candidate!(s, ho.runs, d)
    metadata[:promoted_from] = promoted_from
    return RunEntry(id, _add_r(params, target.resource), unit_params, metadata;
                    pre_artefact=ho.runs[promoted_from].post_artefact)
end

function create_run_entry(s::DEHB, ho, id, params, unit_params)
    de = _subpopulations!(s)
    entry = @invoke create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    r = entry.params.r
    queue = get(de.promotion_queues, r, Int[])
    if haskey(entry.metadata, :promoted_from) && !isempty(queue) && first(queue) == entry.metadata[:promoted_from]
        popfirst!(queue)
    end
    entry.metadata[:slot] = _next_slot(s, r)
    de.total_dispatched[r] += 1
    return entry
end

function on_tell!(s::DEHB, runs, entry)
    de = _subpopulations!(s)
    if entry.status === Completed
        r, slot = entry.params.r, entry.metadata[:slot]
        occupant = de.subpops[r][slot]
        if occupant == 0 || entry.value <= runs[occupant].value
            de.subpops[r][slot] = entry.id
        end
    end
    return @invoke on_tell!(s::SHSync, runs, entry)
end
