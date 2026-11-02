"""
    SobolSampler(; rng=StableRNG(1))

Draw each trial's hyperparameters as the next point of a Sobol sequence, moved by one random offset drawn from `rng` and wrapped around at 1, so successive trials spread over the space more evenly than independent random draws. Like `RandomSampler`, it generates points as they are asked for and needs no fixed number of trials; after any power-of-2 number of trials, each hyperparameter's range is covered evenly.
"""
struct SobolSampler{T<:Random.AbstractRNG} <: Sampler
    rng::T
    sequence::Union{Nothing,Sobol.SobolSeq}
    shift::Vector{Float64}
    started::Base.RefValue{Bool}
end

SobolSampler(; rng::Random.AbstractRNG=StableRNG(1)) = SobolSampler(rng, nothing, Float64[], Ref(false))

init(s::SobolSampler, candidates, n) =
    SobolSampler(s.rng, Sobol.SobolSeq(length(candidates)), rand(s.rng, length(candidates)), Ref(false))

function (s::SobolSampler)(candidates, runs)
    point = s.started[] ? Sobol.next!(s.sequence) : zeros(length(candidates))
    s.started[] = true
    return mod.(point .+ s.shift, 1.0)
end

on_tell!(::SobolSampler, runs, entry) = nothing
exhausted(::SobolSampler, ho) = false  # COV_EXCL_LINE
blocked(::SobolSampler, ho) = false  # COV_EXCL_LINE
create_run_entry(::SobolSampler, ho, id, params, unit_params) = RunEntry(id, params, unit_params)
