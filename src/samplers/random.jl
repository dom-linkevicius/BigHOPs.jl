"""
    RandomSampler(rng=StableRNG(1))

Draw each trial's hyperparameters independently at random. Seeded by default, so repeated searches propose the same trials.
"""
struct RandomSampler{T<:Random.AbstractRNG} <: Sampler
    rng::T
end

RandomSampler() = RandomSampler(StableRNG(1))

function (s::RandomSampler)(candidates, runs)
    return [rand(s.rng) for _ in candidates]
end

on_tell!(::RandomSampler, runs, entry) = nothing
init(s::RandomSampler, candidates, n) = s
exhausted(::RandomSampler, ho) = false
blocked(::RandomSampler, ho) = false
create_run_entry(::RandomSampler, ho, id, params, unit_params) = RunEntry(id, params, unit_params)
