"""
    DEHBSampler(; F=0.5, crossover=0.5, rng=StableRNG(1))

The differential-evolution sampler inside `DEHB`; construct `DEHB` rather than using this directly. `F` scales mutations, and `crossover` is the probability of taking each hyperparameter from the mutant.
"""
struct DEHBSampler{T<:Random.AbstractRNG} <: Sampler
    F::Float64
    crossover::Float64
    rng::T
    subpops::Dict{Int,Vector{Int}}
end
function DEHBSampler(; F::Real=0.5, crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1))
    0 < F <= 1 || throw(ArgumentError("F must be in (0, 1], got $F"))
    0 <= crossover <= 1 || throw(ArgumentError("crossover must be in [0,1], got $crossover"))
    return DEHBSampler(Float64(F), Float64(crossover), rng, Dict{Int,Vector{Int}}())
end

init(s::DEHBSampler, candidates, n) = DEHBSampler(s.F, s.crossover, s.rng, Dict{Int,Vector{Int}}())

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

_subpop_size(s::DEHB, bracket::Int) = _capacity(s.R, s.r_min, s.η, bracket, 1)

function init(s::DEHB, candidates, n)
    sh = @invoke init(s::SuccessiveHalving, candidates, n)
    for bracket in 1:(_smax(s.R, s.r_min, s.η)+1)
        r = _resource(s.R, s.r_min, s.η, bracket, 1)
        sh.inner.subpops[r] = zeros(Int, _subpop_size(s, bracket))
    end
    return sh
end

function _next_slot(s::DEHB, runs, r::Int)
    de = s.inner
    return mod(count(e -> e.params.r == r, runs), length(de.subpops[r])) + 1
end

_ranked(runs, ids) = ids[sortperm([runs[id].value for id in ids])]

function _members(s::DEHB, runs, r::Int, n::Int)
    ranked = _ranked(runs, filter(!=(0), s.inner.subpops[r]))
    members = Vector{Float64}[runs[id].unit_params for id in ranked[1:min(n, end)]]
    return append!(members, [rand(s.inner.rng, length(first(runs).unit_params)) for _ in 1:(n-length(members))])
end

function _global_pool(s::DEHB, runs)
    pool = Vector{Vector{Float64}}()
    for slots in values(s.inner.subpops)
        append!(pool, Vector{Float64}[id == 0 ? rand(s.inner.rng, length(first(runs).unit_params)) : runs[id].unit_params for id in slots])
    end
    return pool
end

function _de_trial(s::DEHBSampler, target::Vector{Float64}, parents::Vector{Vector{Float64}})
    i = findfirst(==(target), parents)
    if i !== nothing && length(parents) > 1
        parents = deleteat!(copy(parents), i)
    end
    parents = vcat(parents, [rand(s.rng, length(target)) for _ in 1:(3-length(parents))])
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

function _promotion_candidate(s::DEHB, runs, d::SHDecision{:promote})
    lower, higher = d.rung.resource, d.bracket.rungs[d.rung.rung+1].resource
    ranked = _ranked(runs, filter(!=(0), s.inner.subpops[lower]))
    run_higher = Set(e.unit_params for e in runs if e.params.r == higher)
    i = findfirst(id -> runs[id].unit_params ∉ run_higher, ranked)
    return i === nothing ? first(ranked) : ranked[i]
end

_propose(d::SHDecision{:promote}, s::DEHB, candidates, runs) =
    d.bracket.iteration == 1 ? copy(runs[_promotion_candidate(s, runs, d)].unit_params) :
    _sample_sh_inner(s, candidates, runs, d)

function _entry_for(d::SHDecision{:promote}, s::DEHB, ho, id, params, unit_params)
    d.bracket.iteration == 1 && return @invoke _entry_for(_promote(d.bracket, d.rung, _promotion_candidate(s, ho.runs, d))::SHDecision{:promote}, s::SuccessiveHalving, ho, id, params, unit_params)
    target = d.bracket.rungs[d.rung.rung+1]
    return RunEntry(id, _add_r(params, target.resource), unit_params, _metadata(d.bracket, target.rung))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:draw})
    r = d.rung.resource
    _init_bracket(d.bracket) && return rand(s.inner.rng, length(candidates))
    return _sample_sh_inner(s, candidates, runs, r,
                            _members(s, runs, r, d.rung.capacity))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:promote})
    r = d.bracket.rungs[d.rung.rung+1].resource
    return _sample_sh_inner(s, candidates, runs, r, _members(s, runs, d.rung.resource, d.bracket.rungs[d.rung.rung+1].capacity))
end

function _sample_sh_inner(s::DEHB, candidates, runs, r::Int, parents)
    de = s.inner

    occupant = de.subpops[r][_next_slot(s, runs, r)]
    pool = _global_pool(s, runs)

    target = occupant == 0 ? rand(de.rng, length(candidates)) : runs[occupant].unit_params
    append!(parents, rand(de.rng, pool, max(0, 3 - length(parents))))
    return _de_trial(de, target, parents)
end

function create_run_entry(s::DEHB, ho, id, params, unit_params)
    entry = @invoke create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    r = entry.params.r
    entry.metadata[:slot] = _next_slot(s, ho.runs, r)
    return entry
end

function on_tell!(s::DEHB, runs, entry)
    de = s.inner
    if entry.status === Completed
        r = entry.params.r
        slot = entry.metadata[:slot]
        occupant = de.subpops[r][slot]
        if occupant == 0 || entry.value <= runs[occupant].value
            de.subpops[r][slot] = entry.id
        end
    end
    return @invoke on_tell!(s::SHSync, runs, entry)
end
