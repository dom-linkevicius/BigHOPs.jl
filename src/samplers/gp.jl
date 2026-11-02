"""
    GPSampler(gp; next_point, noise, n_initial, fit_hyperparams=nothing, standardize=BigHOPs.zscore, rng=StableRNG(1))

Model the objective as a Gaussian process over the unit-encoded hyperparameters, conditioned on every `Completed` trial after each tell, and let `next_point` choose each new trial. `Nominal` hyperparameters enter the kernel through their ordered unit encoding.

- `gp`: an AbstractGPs prior, e.g. `GP(Matern52Kernel())`.
- `next_point(posterior, d, rng)`: return the next trial as a point in `[0, 1]^d`, e.g. `ThompsonSampling`.
- `noise`: the observation noise variance, or `nothing` when `fit_hyperparams` returns one. A noise returned by `fit_hyperparams` replaces it, with a warning the first time if it was a number; the one in use is `ho.sampler.noise[]`.
- `n_initial`: how many trials are drawn uniformly at random before `next_point` is used.
- `fit_hyperparams(X, y)`: return either the GP to condition on, e.g. one with refitted kernel hyperparameters, or a `(gp, noise)` pair to refit the noise variance too. When `nothing`, `gp` is used as given.
- `standardize(y)`: transform the observed values before `fit_hyperparams` and conditioning; the default shifts and scales them to mean 0 and standard deviation 1. When `nothing`, they are used as given.

`next_point`, `fit_hyperparams` and `standardize` are saved in checkpoints by name, so pass named functions or callable structs such as `ThompsonSampling`: an anonymous function is reloaded as whichever function carries its generated name in the new session.

# Examples

Two `fit_hyperparams` that maximize the likelihood on every tell, using Optim and Zygote, which BigHOPs does not depend on. `fit_kernel` refits one length scale per hyperparameter and the kernel variance and returns only a GP, so `noise` stays fixed; `fit_kernel_and_noise` also refits the noise variance and returns a `(gp, noise)` pair, so `noise` can be `nothing`. Both optimize logarithms within bounds, which keeps every value positive and stops the fit collapsing onto degenerate solutions while there are few trials:

```julia
using BigHOPs, AbstractGPs, Optim, Zygote

noise = 1e-4
kernel(lengthscales, variance) = variance * with_lengthscale(Matern52Kernel(), lengthscales)
negative_log_likelihood(X, y, lengthscales, variance, noise) = -logpdf(GP(kernel(lengthscales, variance))(X, noise), y)

function minimize_within(f, lower, upper)
    result = optimize(f, v -> only(Zygote.gradient(f, v)), lower, upper, (lower .+ upper) ./ 2, Fminbox(LBFGS()); inplace=false)
    return Optim.minimizer(result)
end

function fit_kernel(X, y)
    d = length(first(X))
    v = minimize_within(v -> negative_log_likelihood(X, y, exp.(v[1:d]), exp(v[d+1]), noise),
                        fill(log(1e-2), d + 1), fill(log(1e2), d + 1))
    return GP(kernel(exp.(v[1:d]), exp(v[d+1])))
end

function fit_kernel_and_noise(X, y)
    d = length(first(X))
    v = minimize_within(v -> negative_log_likelihood(X, y, exp.(v[1:d]), exp(v[d+1]), exp(v[d+2])),
                        [fill(log(1e-2), d + 1); log(1e-6)], [fill(log(1e2), d + 1); 0.0])
    return GP(kernel(exp.(v[1:d]), exp(v[d+1]))), exp(v[d+2])
end

fixed_noise = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(1000; jitter=1e-6), noise=noise, n_initial=10,
                        fit_hyperparams=fit_kernel)
fitted_noise = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(1000; jitter=1e-6), noise=nothing, n_initial=10,
                         fit_hyperparams=fit_kernel_and_noise)
```
"""
struct GPSampler{T<:Random.AbstractRNG} <: Sampler
    gp::AbstractGP
    next_point::Any
    initial_noise::Union{Nothing,Float64}
    noise::Base.RefValue{Union{Nothing,Float64}}
    n_initial::Int
    fit_hyperparams::Any
    standardize::Any
    rng::T
    posterior::Base.RefValue{AbstractGP}
end

function GPSampler(gp::AbstractGP; next_point, noise::Union{Nothing,Real}, n_initial::Int, fit_hyperparams=nothing, standardize=zscore, rng::Random.AbstractRNG=StableRNG(1))
    noise === nothing || noise >= 0 || throw(ArgumentError("noise must be a non-negative variance, got $noise"))
    noise === nothing && fit_hyperparams === nothing &&
        throw(ArgumentError("noise can only be nothing when fit_hyperparams returns a (gp, noise) pair"))
    n_initial >= 0 || throw(ArgumentError("n_initial must be >= 0, got $n_initial"))
    return GPSampler(gp, next_point, noise, Ref{Union{Nothing,Float64}}(noise), n_initial, fit_hyperparams, standardize, rng, Ref{AbstractGP}(gp))
end

function (s::GPSampler)(candidates, runs)
    length(runs) < s.n_initial && return rand(s.rng, length(candidates))
    return s.next_point(s.posterior[], length(candidates), s.rng)
end

function zscore(y)
    spread = std(y)
    return spread > 0 ? (y .- mean(y)) ./ spread : y .- mean(y)
end

function _fitted!(s::GPSampler, gp::AbstractGP)
    s.noise[] === nothing &&
        throw(ArgumentError("fit_hyperparams returned only a GP, but noise is nothing -- return a (gp, noise) pair or pass a noise"))
    return gp
end

function _fitted!(s::GPSampler, (gp, noise)::Tuple{AbstractGP,Real})
    noise >= 0 || throw(ArgumentError("fit_hyperparams returned a negative noise variance, $noise"))
    s.initial_noise === nothing || @warn "noise=$(s.initial_noise) is replaced by the noise fit_hyperparams returns" maxlog = 1 _id = objectid(s.noise)
    s.noise[] = noise
    return gp
end

_fitted!(::GPSampler, returned) =
    throw(ArgumentError("fit_hyperparams must return a GP or a (gp, noise) pair, got a $(typeof(returned))"))

function on_tell!(s::GPSampler, runs, entry::RunEntry)
    entry.status === Completed || return nothing
    done = filter(e -> e.status === Completed, runs)
    X = ColVecs(stack(e.unit_params for e in done))
    observed = Float64[e.value for e in done]
    y = s.standardize === nothing ? observed : s.standardize(observed)
    gp = s.fit_hyperparams === nothing ? s.gp : _fitted!(s, s.fit_hyperparams(X, y))
    s.posterior[] = posterior(gp(X, s.noise[]), y)
    return nothing
end

init(s::GPSampler, candidates, n) = GPSampler(s.gp, s.next_point, s.initial_noise, Ref{Union{Nothing,Float64}}(s.initial_noise), s.n_initial, s.fit_hyperparams, s.standardize, s.rng, Ref{AbstractGP}(s.gp))
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
