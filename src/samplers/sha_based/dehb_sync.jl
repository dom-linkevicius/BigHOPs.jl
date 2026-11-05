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

init(s::DEHBSampler, candidates, n) = s

_dehb_standalone() = throw(ArgumentError("DEHBSampler can't be used on its own; construct DEHB(; R, ...) instead"))
on_tell!(::DEHBSampler, runs, entry) = _dehb_standalone()
exhausted(::DEHBSampler, ho) = _dehb_standalone()
blocked(::DEHBSampler, ho) = _dehb_standalone()
create_run_entry(::DEHBSampler, ho, id, params, unit_params) = _dehb_standalone()
(::DEHBSampler)(candidates, runs) = _dehb_standalone()

"""
    DEHB(; R, η=3, r_min=1, iterations=1, F=0.5, crossover=0.5, rng=StableRNG(1))

`Hyperband`'s schedule with differential evolution in place of random draws, following the
paper (Awad et al., 2021, https://doi.org/10.24963/ijcai.2021/296), with the details it
leaves open taken from the reference implementation (https://github.com/automl/DEHB).

The first rung of the first bracket is drawn uniformly at random, and every other first-rung 
trial is evolved from its own level's subpopulation, with a random target standing in 
for a slot that is still empty. In the first iteration, later rungs promote the best configurations 
from the level below unchanged; in later iterations every trial is evolved from the level below. 
Every trial trains from scratch: a `Stateful` objective never receives a `pre_artefact` 
and only has its `post_artefact` recorded.
"""
const DEHB = SuccessiveHalving{true,<:DEHBSampler}

SuccessiveHalving{true,<:DEHBSampler}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1, F::Real=0.5,
                                       crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1)) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, iterations=iterations,
                            inner=DEHBSampler(; F=F, crossover=crossover, rng=rng))

_check_objective(s::DEHB, objective) = objective isa Stateful && @warn "$(typeof(s)) with a Stateful objective: DEHB never sets pre_artefact, so every trial trains from scratch; Stateful only records post_artefact"

function _init_inner(inner::DEHBSampler, s::SuccessiveHalving, candidates)
    de = DEHBSampler(inner.F, inner.crossover, inner.rng, Dict{Int,Vector{Int}}())
    for bracket in 1:(_smax(s.R, s.r_min, s.η)+1)
        r = _resource(s.R, s.r_min, s.η, bracket, 1)
        de.subpops[r] = zeros(Int, _capacity(s.R, s.r_min, s.η, bracket, 1))
    end
    return de
end

_slot(s::DEHB, runs, r::Int, id::Int) = mod(count(e -> e.params.r == r && e.id < id, runs), length(s.inner.subpops[r])) + 1

_ranked(runs, slots) = sort(filter(!=(0), slots); by=id -> runs[id].value)

function _members(s::DEHB, runs, r::Int, n::Int)
    ranked = _ranked(runs, s.inner.subpops[r])
    members = Vector{Float64}[runs[id].unit_params for id in ranked[1:min(n, end)]]
    return append!(members, [rand(s.inner.rng, length(first(runs).unit_params)) for _ in 1:(n-length(members))])
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

function _promotion_candidate(s::DEHB, runs, d::SHDecision{:promote})
    lower, higher = d.rung.resource, d.bracket.rungs[d.rung.rung+1].resource
    ranked = _ranked(runs, s.inner.subpops[lower])
    run_higher = Set(e.unit_params for e in runs if e.params.r == higher)
    return ranked[something(findfirst(id -> runs[id].unit_params ∉ run_higher, ranked), 1)]
end

_propose(d::SHDecision{:promote}, s::DEHB, candidates, runs) =
    d.bracket.iteration == 1 ? copy(runs[_promotion_candidate(s, runs, d)].unit_params) :
    _sample_sh_inner(s, candidates, runs, d)

function _entry_for(d::SHDecision{:promote}, s::DEHB, ho, id, params, unit_params)
    target = d.bracket.rungs[d.rung.rung+1]
    metadata = _metadata(d.bracket, target.rung)
    if d.bracket.iteration == 1
        metadata[:promoted_from] = _promotion_candidate(s, ho.runs, d)
    end
    return RunEntry(id, _add_r(params, target.resource), unit_params, metadata)
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:draw})
    r = d.rung.resource
    d.bracket.iteration == 1 && d.bracket.bracket == 1 && return rand(s.inner.rng, length(candidates))
    return _sample_sh_inner(s, candidates, runs, r,
                            _members(s, runs, r, d.rung.capacity))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:promote})
    r = d.bracket.rungs[d.rung.rung+1].resource
    return _sample_sh_inner(s, candidates, runs, r, _members(s, runs, d.rung.resource, d.bracket.rungs[d.rung.rung+1].capacity))
end

function _sample_sh_inner(s::DEHB, candidates, runs, r::Int, parents)
    de = s.inner

    occupant = de.subpops[r][_slot(s, runs, r, length(runs) + 1)]
    target = occupant == 0 ? rand(de.rng, length(candidates)) : runs[occupant].unit_params
    i = findfirst(==(target), parents)
    if i !== nothing && length(parents) > 1
        deleteat!(parents, i)
    end
    if length(parents) < 3
        global_pool = reduce(vcat, values(de.subpops))
        for id in rand(de.rng, global_pool, 3 - length(parents))
            push!(parents, id == 0 ? rand(de.rng, length(candidates)) : runs[id].unit_params)
        end
    end
    return _de_trial(de, target, parents)
end

function _tell_inner!(de::DEHBSampler, s, runs, entry)
    if entry.status === Completed
        r = entry.params.r
        slot = _slot(s, runs, r, entry.id)
        occupant = de.subpops[r][slot]
        if occupant == 0 || entry.value <= runs[occupant].value
            de.subpops[r][slot] = entry.id
        end
    end
    return nothing
end
