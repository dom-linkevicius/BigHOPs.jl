"""
    GPSampler(gp; next_point, noise, n_initial, fit_hyperparams=nothing, standardize=BigHOPs.zscore, strict_initialization=true, rng=StableRNG(1))

Model the objective as a Gaussian process over the unit-encoded hyperparameters, conditioned on every `Completed` trial after each tell, and let `next_point` choose each new trial. `Nominal` hyperparameters enter the kernel through their ordered unit encoding.

- `gp`: an AbstractGPs prior, e.g. `GP(Matern52Kernel())`.
- `next_point(posterior, d, rng)`: return the next trial as a point in `[0, 1]^d`, e.g. `ThompsonSampling`.
- `noise`: the observation noise variance, or `nothing` when `fit_hyperparams` returns one. A noise returned by `fit_hyperparams` replaces it, with a warning the first time if it was a number; the one in use is `ho.sampler.noise[]`.
- `n_initial`: how many trials are drawn uniformly at random before `next_point` is used. No further trial is proposed until all of them have finished; `next_point` then takes over from those that completed, with a warning if that is fewer than `n_initial`.
- `fit_hyperparams(X, y)`: return either the GP to condition on, e.g. one with refitted kernel hyperparameters, or a `(gp, noise)` pair to refit the noise variance too. When `nothing`, `gp` is used as given.
- `standardize(y)`: transform the observed values before `fit_hyperparams` and conditioning; the default shifts and scales them to mean 0 and standard deviation 1. When `nothing`, they are used as given.
- `strict_initialization`: when `false`, each initial trial that does not complete gets an extra random trial instead, with a warning, until `n_initial` have completed; the unsuccessful ones stay in the results, and new proposals wait only while the initial trials still running could complete the set.

`next_point`, `fit_hyperparams` and `standardize` are saved in checkpoints by name, so pass named functions or callable structs such as `ThompsonSampling`: an anonymous function is reloaded as whichever function carries its generated name in the new session.

# Examples

Both examples maximize the likelihood on every tell using Optim and Zygote, which BigHOPs does not depend on. They optimize logarithms within bounds, which keeps every value positive and stops the fit collapsing onto degenerate solutions while there are few trials.

```julia
# Refits the kernel hyperparameters only; fit_kernel returns a GP, so the noise stays fixed at `noise`.
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

sampler = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(10; jitter=1e-6), noise=noise, n_initial=10,
                    fit_hyperparams=fit_kernel)
```

```julia
# Refits the kernel hyperparameters and the noise variance; fit_kernel_and_noise returns a (gp, noise) pair, so `noise` can be nothing.
using BigHOPs, AbstractGPs, Optim, Zygote

kernel(lengthscales, variance) = variance * with_lengthscale(Matern52Kernel(), lengthscales)
negative_log_likelihood(X, y, lengthscales, variance, noise) = -logpdf(GP(kernel(lengthscales, variance))(X, noise), y)

function minimize_within(f, lower, upper)
    result = optimize(f, v -> only(Zygote.gradient(f, v)), lower, upper, (lower .+ upper) ./ 2, Fminbox(LBFGS()); inplace=false)
    return Optim.minimizer(result)
end

function fit_kernel_and_noise(X, y)
    d = length(first(X))
    v = minimize_within(v -> negative_log_likelihood(X, y, exp.(v[1:d]), exp(v[d+1]), exp(v[d+2])),
                        [fill(log(1e-2), d + 1); log(1e-6)], [fill(log(1e2), d + 1); 0.0])
    return GP(kernel(exp.(v[1:d]), exp(v[d+1]))), exp(v[d+2])
end

sampler = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(10; jitter=1e-6), noise=nothing, n_initial=10,
                    fit_hyperparams=fit_kernel_and_noise)
```
"""
struct GPSampler{T<:Random.AbstractRNG} <: Sampler
    gp::AbstractGP
    next_point::Any
    noise::Base.RefValue{Union{Nothing,Float64}}
    n_initial::Int
    fit_hyperparams::Any
    standardize::Any
    strict_initialization::Bool
    rng::T
    posterior::Base.RefValue{Union{Nothing,AbstractGP}}
end

function GPSampler(gp::AbstractGP; next_point, noise::Union{Nothing,Real}, n_initial::Int, fit_hyperparams=nothing, standardize=zscore, strict_initialization::Bool=true, rng::Random.AbstractRNG=StableRNG(1))
    noise === nothing || noise >= 0 || throw(ArgumentError("noise must be a non-negative variance, got $noise"))
    noise === nothing && fit_hyperparams === nothing &&
        throw(ArgumentError("noise can only be nothing when fit_hyperparams returns a (gp, noise) pair"))
    n_initial >= 0 || throw(ArgumentError("n_initial must be >= 0, got $n_initial"))
    return GPSampler(gp, next_point, Ref{Union{Nothing,Float64}}(noise), n_initial, fit_hyperparams, standardize, strict_initialization, rng, Ref{Union{Nothing,AbstractGP}}(nothing))
end

function (s::GPSampler)(candidates, runs)
    length(runs) < s.n_initial && return rand(s.rng, length(candidates))
    completed = count(e -> e.status === Completed, runs)
    if s.strict_initialization
        length(runs) == s.n_initial && completed < s.n_initial &&
            @warn "only $completed of the $(s.n_initial) initial trials completed; strict_initialization is set, so next_point continues from those"
    elseif completed < s.n_initial
        @warn "only $completed of the $(s.n_initial) initial trials have completed; drawing an extra random trial"
        return rand(s.rng, length(candidates))
    end
    return s.next_point(something(s.posterior[], s.gp), length(candidates), s.rng)
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
    s.posterior[] === nothing && s.noise[] !== nothing && @warn "noise=$(s.noise[]) is replaced by the noise fit_hyperparams returns"
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

init(s::GPSampler, candidates, n) = GPSampler(s.gp, s.next_point, Ref{Union{Nothing,Float64}}(s.noise[]), s.n_initial, s.fit_hyperparams, s.standardize, s.strict_initialization, s.rng, Ref{Union{Nothing,AbstractGP}}(nothing))
exhausted(::GPSampler, ho) = false  # COV_EXCL_LINE
function blocked(s::GPSampler, ho)
    s.strict_initialization && return length(ho.runs) >= s.n_initial && any(e -> e.status === Pending, view(ho.runs, 1:s.n_initial))
    return length(ho.completed) < s.n_initial <= length(ho.completed) + ho.n_pending
end
create_run_entry(::GPSampler, ho, id, params, unit_params) = RunEntry(id, params, unit_params)

"""
    ThompsonSampling(log2_candidates; jitter)

A `next_point` for `GPSampler`: draw the GP posterior jointly at `2^log2_candidates` candidate points and return the one where the draw is lowest (Kandasamy et al., 2018, https://proceedings.mlr.press/v84/kandasamy18a.html). The candidates are a Sobol sequence scrambled afresh from `rng` on every call; the scrambling (Owen's) needs a power-of-2 number of points, hence the exponent, e.g. `10` for 1024 candidates. `jitter` is the variance added to the draw's covariance so that it can be factorized.
"""
struct ThompsonSampling
    log2_candidates::Int
    jitter::Float64

    function ThompsonSampling(log2_candidates::Int, jitter::Float64)
        log2_candidates >= 1 || throw(ArgumentError("log2_candidates must be >= 1, got $log2_candidates"))
        jitter >= 0 || throw(ArgumentError("jitter must be a non-negative variance, got $jitter"))
        return new(log2_candidates, jitter)
    end
end

ThompsonSampling(log2_candidates::Int; jitter::Real) = ThompsonSampling(log2_candidates, Float64(jitter))

function (ts::ThompsonSampling)(post, d::Int, rng)
    scrambled_sobol = QuasiMonteCarlo.SobolSample(R=QuasiMonteCarlo.OwenScramble(base=2, rng=rng))
    candidates = QuasiMonteCarlo.sample(2^ts.log2_candidates, d, scrambled_sobol)
    draw = rand(rng, post(ColVecs(candidates), ts.jitter))
    return candidates[:, argmin(draw)]
end
