"""
    GPSampler(gp; next_point, noise, n_initial, fit=nothing, standardize=BigHOPs.zscore, rng=StableRNG(1))

Model the objective as a Gaussian process over the unit-encoded hyperparameters, conditioned on every `Completed` trial after each tell, and let `next_point` choose each new trial. `Nominal` hyperparameters enter the kernel through their ordered unit encoding.

- `gp`: an AbstractGPs prior, e.g. `GP(Matern52Kernel())`.
- `next_point(posterior, d, rng)`: return the next trial as a point in `[0, 1]^d`, e.g. `ThompsonSampling`.
- `noise`: the observation noise variance.
- `n_initial`: how many trials are drawn uniformly at random before `next_point` is used.
- `fit(X, y)`: return the GP to condition on, e.g. one with refitted kernel hyperparameters. When `nothing`, `gp` is used as given.
- `standardize(y)`: transform the observed values before `fit` and conditioning; the default shifts and scales them to mean 0 and standard deviation 1. When `nothing`, they are used as given.
"""
struct GPSampler{T<:Random.AbstractRNG} <: Sampler
    gp::AbstractGP
    next_point::Any
    noise::Float64
    n_initial::Int
    fit::Any
    standardize::Any
    rng::T
    posterior::Base.RefValue{AbstractGP}
end

function GPSampler(gp::AbstractGP; next_point, noise::Real, n_initial::Int, fit=nothing, standardize=zscore, rng::Random.AbstractRNG=StableRNG(1))
    noise >= 0 || throw(ArgumentError("noise must be a non-negative variance, got $noise"))
    n_initial >= 0 || throw(ArgumentError("n_initial must be >= 0, got $n_initial"))
    return GPSampler(gp, next_point, Float64(noise), n_initial, fit, standardize, rng, Ref{AbstractGP}(gp))
end

function (s::GPSampler)(candidates, runs)
    length(runs) < s.n_initial && return rand(s.rng, length(candidates))
    return s.next_point(s.posterior[], length(candidates), s.rng)
end

function zscore(y)
    spread = std(y)
    return spread > 0 ? (y .- mean(y)) ./ spread : y .- mean(y)
end

function on_tell!(s::GPSampler, runs, entry::RunEntry)
    entry.status === Completed || return nothing
    done = filter(e -> e.status === Completed, runs)
    X = ColVecs(stack(e.unit_params for e in done))
    observed = Float64[e.value for e in done]
    y = s.standardize === nothing ? observed : s.standardize(observed)
    gp = s.fit === nothing ? s.gp : s.fit(X, y)
    s.posterior[] = posterior(gp(X, s.noise), y)
    return nothing
end

init(s::GPSampler, candidates, n) = GPSampler(s.gp, s.next_point, s.noise, s.n_initial, s.fit, s.standardize, s.rng, Ref{AbstractGP}(s.gp))
exhausted(::GPSampler, ho) = false  # COV_EXCL_LINE
blocked(::GPSampler, ho) = false  # COV_EXCL_LINE
create_run_entry(::GPSampler, ho, id, params, unit_params) = RunEntry(id, params, unit_params)

"""
    ThompsonSampling(n_candidates; jitter)

A `next_point` for `GPSampler`: draw the GP posterior jointly at `n_candidates` uniformly random points and return the one where the draw is lowest (Kandasamy et al., 2018, https://proceedings.mlr.press/v84/kandasamy18a.html). `jitter` is the variance added to the draw's covariance so that it can be factorized.
"""
struct ThompsonSampling
    n_candidates::Int
    jitter::Float64

    function ThompsonSampling(n_candidates::Int, jitter::Float64)
        n_candidates >= 1 || throw(ArgumentError("n_candidates must be >= 1, got $n_candidates"))
        jitter >= 0 || throw(ArgumentError("jitter must be a non-negative variance, got $jitter"))
        return new(n_candidates, jitter)
    end
end

ThompsonSampling(n_candidates::Int; jitter::Real) = ThompsonSampling(n_candidates, Float64(jitter))

function (ts::ThompsonSampling)(post, d::Int, rng)
    candidates = rand(rng, d, ts.n_candidates)
    draw = rand(rng, post(ColVecs(candidates), ts.jitter))
    return candidates[:, argmin(draw)]
end
