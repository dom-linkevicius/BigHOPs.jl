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
end
function DEHBSampler(; F::Real=0.5, crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1))
    0 < F <= 1 || throw(ArgumentError("F must be in (0, 1], got $F"))
    0 <= crossover <= 1 || throw(ArgumentError("crossover must be in [0,1], got $crossover"))
    return DEHBSampler(Float64(F), Float64(crossover), rng, Dict{Int,Vector{Int}}(), Dict{Int,Int}())
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

`Hyperband`'s schedule with differential evolution in place of random draws (Awad et al., 2021, https://doi.org/10.24963/ijcai.2021/296). The first bracket runs as plain Hyperband to seed one subpopulation per resource level; after that every trial is evolved from those subpopulations rather than promoted unchanged, so it trains from scratch, and a `Stateful` objective only receives a `pre_artefact` in that first bracket.
"""
const DEHB = SuccessiveHalving{true,<:DEHBSampler}

SuccessiveHalving{true,<:DEHBSampler}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1, F::Real=0.5,
                                       crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1)) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, iterations=iterations,
                            inner=DEHBSampler(; F=F, crossover=crossover, rng=rng))

_check_objective(s::DEHB, objective) = objective isa Stateful && @warn "$(typeof(s)) with a Stateful objective: pre_artefact is only set on promotions in the first bracket of the first iteration; every other trial trains from scratch. post_artefact is recorded either way"

_subpop_size(s::DEHB, bracket::Int) = _capacity(s.R, s.r_min, s.η, bracket, 1)

function _subpopulations!(s::DEHB)
    de = s.inner
    isempty(de.total_dispatched) || return de
    for bracket in 1:(_smax(s.R, s.r_min, s.η)+1)
        r = _resource(s.R, s.r_min, s.η, bracket, 1)
        de.subpops[r] = zeros(Int, _subpop_size(s, bracket))
        de.total_dispatched[r] = 0
    end
    return de
end

function _next_slot(s::DEHB, r::Int)
    de = _subpopulations!(s)
    return mod(de.total_dispatched[r], length(de.subpops[r])) + 1
end

_occupants(runs, slots) = Vector{Float64}[runs[id].unit_params for id in slots if id != 0]

_parent_pool(s::DEHB, runs, r::Int) = _occupants(runs, _subpopulations!(s).subpops[r÷s.η])

function _global_pool(s::DEHB, runs)
    pool = Vector{Vector{Float64}}()
    for slots in values(_subpopulations!(s).subpops)
        append!(pool, _occupants(runs, slots))
    end
    return pool
end

function _de_trial(s::DEHBSampler, target::Vector{Float64}, parents::Vector{Vector{Float64}})
    r1, r2, r3 = parents[randperm(s.rng, length(parents))[1:3]]
    mutant = r1 .+ s.F .* (r2 .- r3)
    out_of_range = .!(0 .<= mutant .<= 1)
    mutant[out_of_range] .= rand(s.rng, count(out_of_range))
    forced = rand(s.rng, eachindex(target))

    cross = rand(s.rng, length(target)) .< s.crossover
    cross[forced] = 1
    return ifelse.(cross, mutant, target)
end

_init_bracket(bracket::ActiveBracket) = bracket.iteration == 1 && bracket.bracket == 1

_propose(d::SHDecision{:promote}, s::DEHB, candidates, runs) =
    _init_bracket(d.bracket) ? copy(runs[d.promoted_from].unit_params) :
    _sample_sh_inner(s, candidates, runs, d)

function _entry_for(d::SHDecision{:promote}, s::DEHB, ho, id, params, unit_params)
    _init_bracket(d.bracket) && return @invoke _entry_for(d::SHDecision{:promote}, s::SuccessiveHalving, ho, id, params, unit_params)
    target = d.bracket.rungs[d.rung.rung+1]
    return RunEntry(id, _add_r(params, target.resource), unit_params, _metadata(d.bracket, target.rung))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:draw})
    r = d.rung.resource
    return _sample_sh_inner(s, candidates, runs, d.bracket, r,
                            _occupants(runs, _subpopulations!(s).subpops[r]))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:promote})
    r = d.bracket.rungs[d.rung.rung+1].resource
    return _sample_sh_inner(s, candidates, runs, d.bracket, r, _parent_pool(s, runs, r))
end

function _sample_sh_inner(s::DEHB, candidates, runs, bracket::ActiveBracket, r::Int, parents)
    de = _subpopulations!(s)
    _init_bracket(bracket) && r == s.r_min && return rand(de.rng, length(candidates))

    occupant = de.subpops[r][_next_slot(s, r)]
    pool = _global_pool(s, runs)

    ### the paper does not seem to mention what happens in the case where there aren't enough targets
    ### whether to take from the same subpopulation, but not replace them or from global population
    target = occupant == 0 ? rand(de.rng, pool) : runs[occupant].unit_params
    append!(parents, rand(de.rng, pool, max(0, 3 - length(parents))))
    return _de_trial(de, target, parents)
end

function create_run_entry(s::DEHB, ho, id, params, unit_params)
    de = _subpopulations!(s)
    entry = @invoke create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    r = entry.params.r
    entry.metadata[:slot] = _next_slot(s, r)
    de.total_dispatched[r] += 1
    return entry
end

function on_tell!(s::DEHB, runs, entry)
    de = _subpopulations!(s)
    if entry.status === Completed
        r = entry.params.r
        slot = entry.metadata[:slot]
        occupant = de.subpops[r][slot]
        if occupant == 0 || entry.value < runs[occupant].value
            de.subpops[r][slot] = entry.id
        end
    end
    return @invoke on_tell!(s::SHSync, runs, entry)
end
