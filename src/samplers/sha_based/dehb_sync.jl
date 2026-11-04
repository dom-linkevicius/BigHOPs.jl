"""
    DEHBSampler(; F=0.5, crossover=0.5, rng=StableRNG(1))

The differential-evolution sampler inside `DEHB`; construct `DEHB` rather than using this directly. `F` scales mutations, and `crossover` is the probability of taking each hyperparameter from the mutant.
"""
struct DEHBSampler{T<:Random.AbstractRNG} <: Sampler
    F::Float64
    crossover::Float64
    rng::T
    subpops::Dict{Int,Vector{Vector{Float64}}}
    fitness::Dict{Int,Vector{Float64}}
    ids::Dict{Int,Vector{Int}}
    total_dispatched::Dict{Int,Int}
    promotion_queues::Dict{Int,Vector{Int}}
end
function DEHBSampler(; F::Real=0.5, crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1))
    0 < F <= 1 || throw(ArgumentError("F must be in (0, 1], got $F"))
    0 <= crossover <= 1 || throw(ArgumentError("crossover must be in [0,1], got $crossover"))
    return DEHBSampler(Float64(F), Float64(crossover), rng, Dict{Int,Vector{Vector{Float64}}}(),
                       Dict{Int,Vector{Float64}}(), Dict{Int,Vector{Int}}(), Dict{Int,Int}(), Dict{Int,Vector{Int}}())
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

`Hyperband`'s schedule with differential evolution in place of random draws, following the reference implementation (Awad et al., 2021, https://doi.org/10.24963/ijcai.2021/296, https://github.com/automl/DEHB). Each resource level keeps a subpopulation, seeded with random configurations, and every first-rung trial is evolved from its own level's subpopulation. In the first iteration, later rungs promote the best configurations from the level below unchanged, so a `Stateful` objective receives their `pre_artefact`; in later iterations every trial is evolved from the level below and trains from scratch.
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

function _subpopulations!(s::DEHB, d::Int)
    de = s.inner
    isempty(de.total_dispatched) || return de
    for bracket in 1:(_smax(s.R, s.r_min, s.η)+1)
        r = _resource(s.R, s.r_min, s.η, bracket, 1)
        n = _subpop_size(s, r)
        de.subpops[r] = [rand(de.rng, d) for _ in 1:n]
        de.fitness[r] = fill(Inf, n)
        de.ids[r] = zeros(Int, n)
        de.total_dispatched[r] = 0
    end
    return de
end

_next_slot(s::DEHB, r::Int) = mod(s.inner.total_dispatched[r], length(s.inner.subpops[r])) + 1

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

_mutation_pool(de::DEHBSampler, r::Int, n::Int) = de.subpops[r][sortperm(de.fitness[r])[1:min(n, end)]]

function _evolve(s::DEHB, d::Int, r::Int, r_parents::Int, n::Int)
    de = _subpopulations!(s, d)
    target = de.subpops[r][_next_slot(s, r)]
    pool = _mutation_pool(de, r_parents, n)
    if length(pool) < 3
        everyone = reduce(vcat, (de.subpops[level] for level in sort(collect(keys(de.subpops)))))
        append!(pool, [_mutant(de, target, everyone) for _ in 1:(4-length(pool))])
    end
    return _de_trial(de, target, pool)
end

function _promotion_candidate!(s::DEHB, d::Int, decision::SHDecision{:promote})
    de = _subpopulations!(s, d)
    lower, higher = decision.rung, decision.bracket.rungs[decision.rung.rung+1]
    evaluated = findall(isfinite, de.fitness[lower.resource])
    ranked = evaluated[sortperm(de.fitness[lower.resource][evaluated])]
    queue = get!(de.promotion_queues, higher.resource, Int[])
    if isempty(queue)
        append!(queue, [de.ids[lower.resource][i] for i in ranked if de.subpops[lower.resource][i] ∉ de.subpops[higher.resource]])
        resize!(queue, min(length(queue), higher.capacity))
    end
    return isempty(queue) ? de.ids[lower.resource][first(ranked)] : first(queue)
end

_sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:draw}) =
    _evolve(s, length(candidates), d.rung.resource, d.rung.resource, d.rung.capacity)

function _propose(d::SHDecision{:promote}, s::DEHB, candidates, runs)
    _promotion_phase(d.bracket) && return copy(runs[_promotion_candidate!(s, length(candidates), d)].unit_params)
    higher = d.bracket.rungs[d.rung.rung+1]
    return _evolve(s, length(candidates), higher.resource, d.rung.resource, higher.capacity)
end

function _entry_for(d::SHDecision{:promote}, s::DEHB, ho, id, params, unit_params)
    target = d.bracket.rungs[d.rung.rung+1]
    metadata = _metadata(d.bracket, target.rung)
    _promotion_phase(d.bracket) || return RunEntry(id, _add_r(params, target.resource), unit_params, metadata)
    promoted_from = _promotion_candidate!(s, length(unit_params), d)
    metadata[:promoted_from] = promoted_from
    return RunEntry(id, _add_r(params, target.resource), unit_params, metadata;
                    pre_artefact=ho.runs[promoted_from].post_artefact)
end

function create_run_entry(s::DEHB, ho, id, params, unit_params)
    de = _subpopulations!(s, length(unit_params))
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
    de = _subpopulations!(s, length(entry.unit_params))
    if entry.status === Completed
        r, slot = entry.params.r, entry.metadata[:slot]
        if entry.value <= de.fitness[r][slot]
            de.subpops[r][slot] = copy(entry.unit_params)
            de.fitness[r][slot] = entry.value
            de.ids[r][slot] = entry.id
        end
    end
    return @invoke on_tell!(s::SHSync, runs, entry)
end
