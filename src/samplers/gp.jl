"""
    GPSampler(gp; next_point, noise, n_initial, initial=SobolSampler(), fit_hyperparams=nothing, standardize=BigHOPs.zscore, strict_initialization=true, rng=StableRNG(1))

Model the objective as a Gaussian process over the unit-encoded hyperparameters and let `next_point` choose each new trial once the initial trials are done. From then on, the GP is conditioned on every `Completed` trial after each tell; during the initial trials it is not. `Nominal` hyperparameters enter the kernel through their ordered unit encoding.

- `gp`: an AbstractGPs prior, e.g. `GP(Matern52Kernel())`.
- `next_point(posterior, d, rng)`: return the next trial as a point in `[0, 1]^d`, e.g. `ThompsonSampling`.
- `noise`: the observation noise variance, or `nothing` when `fit_hyperparams` returns one. A noise returned by `fit_hyperparams` replaces it, with a warning the first time if it was a number; the one in use is `ho.sampler.noise[]`.
- `n_initial`: how many trials `initial` draws before `next_point` is used. No further trial is proposed until all of them have finished; `next_point` then takes over from those that completed, with a warning if that is fewer than `n_initial`.
- `initial`: the sampler that draws the initial trials, `SobolSampler` by default, or `RandomSampler`, or `LHSampler` when `strict_initialization` is `true`.
- `fit_hyperparams(X, y)`: called before each conditioning; return either the GP to condition on, e.g. one with refitted kernel hyperparameters, or a `(gp, noise)` pair to refit the noise variance too. When `nothing`, `gp` is used as given.
- `standardize(y)`: transform the observed values before `fit_hyperparams` and conditioning; the default shifts and scales them to mean 0 and standard deviation 1. When `nothing`, they are used as given.
- `strict_initialization`: when `false`, each initial trial that does not complete gets an extra trial from `initial` instead, with a warning, until `n_initial` have completed; the unsuccessful ones stay in the results, and new proposals wait only while the initial trials still running could complete the set.

`next_point`, `fit_hyperparams` and `standardize` are saved in checkpoints by name, so pass named functions or callable structs such as `ThompsonSampling`.
"""
struct GPSampler{T<:Random.AbstractRNG} <: Sampler
    gp::AbstractGP
    next_point::Any
    noise::Base.RefValue{Union{Nothing,Float64}}
    n_initial::Int
    initial::BasicSamplers
    fit_hyperparams::Any
    standardize::Any
    strict_initialization::Bool
    rng::T
    posterior::Base.RefValue{Union{Nothing,AbstractGP}}
end

function GPSampler(gp::AbstractGP; next_point, noise::Union{Nothing,Real}, n_initial::Int, initial::BasicSamplers=SobolSampler(), fit_hyperparams=nothing, standardize=zscore, strict_initialization::Bool=true, rng::Random.AbstractRNG=StableRNG(1))
    noise === nothing || noise >= 0 || throw(ArgumentError("noise must be a non-negative variance, got $noise"))
    noise === nothing && fit_hyperparams === nothing &&
        throw(ArgumentError("noise can only be nothing when fit_hyperparams returns a (gp, noise) pair"))
    n_initial >= 0 || throw(ArgumentError("n_initial must be >= 0, got $n_initial"))
    initial isa LHSampler && !strict_initialization &&
        throw(ArgumentError("an LHSampler draws exactly n_initial points, so it cannot draw extra initial trials -- use strict_initialization=true, or a SobolSampler or RandomSampler"))
    return GPSampler(gp, next_point, Ref{Union{Nothing,Float64}}(noise), n_initial, initial, fit_hyperparams, standardize, strict_initialization, rng, Ref{Union{Nothing,AbstractGP}}(nothing))
end

function (s::GPSampler)(candidates, runs)
    length(runs) < s.n_initial && return s.initial(candidates, runs)
    completed = count(e -> e.status === Completed, runs)
    if s.strict_initialization
        length(runs) == s.n_initial && completed < s.n_initial &&
            @warn "only $completed of the $(s.n_initial) initial trials completed; strict_initialization is set, so next_point continues from those"
    elseif completed < s.n_initial
        @warn "only $completed of the $(s.n_initial) initial trials have completed; drawing an extra initial trial"
        return s.initial(candidates, runs)
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

function _initializing(s::GPSampler, runs)
    s.strict_initialization || return count(e -> e.status === Completed, runs) < s.n_initial
    return length(runs) < s.n_initial || any(e -> e.status === Pending, view(runs, 1:s.n_initial))
end

function on_tell!(s::GPSampler, runs, entry::RunEntry)
    _initializing(s, runs) && return nothing
    entry.status === Completed || s.posterior[] === nothing || return nothing
    done = filter(e -> e.status === Completed, runs)
    isempty(done) && return nothing
    X = ColVecs(stack(e.unit_params for e in done))
    observed = Float64[e.value for e in done]
    y = s.standardize === nothing ? observed : s.standardize(observed)
    gp = s.fit_hyperparams === nothing ? s.gp : _fitted!(s, s.fit_hyperparams(X, y))
    s.posterior[] = posterior(gp(X, s.noise[]), y)
    return nothing
end

function init(s::GPSampler, candidates, n)
    initial = s.n_initial == 0 ? s.initial : init(s.initial, candidates, s.n_initial)
    return GPSampler(s.gp, s.next_point, Ref{Union{Nothing,Float64}}(s.noise[]), s.n_initial, initial, s.fit_hyperparams, s.standardize, s.strict_initialization, s.rng, Ref{Union{Nothing,AbstractGP}}(nothing))
end

exhausted(::GPSampler, ho) = false  # COV_EXCL_LINE
function blocked(s::GPSampler, ho)
    s.strict_initialization && return length(ho.runs) >= s.n_initial && any(e -> e.status === Pending, view(ho.runs, 1:s.n_initial))
    completed = length(ho.completed)
    return completed < s.n_initial && completed + ho.n_pending >= s.n_initial
end
create_run_entry(::GPSampler, ho, id, params, unit_params) = RunEntry(id, params, unit_params)

"""
    ThompsonSampling(n_candidates; jitter)

A `next_point` for `GPSampler`: draw the GP posterior jointly at `n_candidates` candidate points and return the one where the draw is lowest (Kandasamy et al., 2018, https://proceedings.mlr.press/v84/kandasamy18a.html). The candidates are Sobol points moved by one random offset, drawn afresh from `rng` on every call, and wrapped around at 1; a power-of-2 number of them covers each hyperparameter exactly evenly. `jitter` is the variance added to the draw's covariance so that it can be factorized.
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
    shifted_sobol = QuasiMonteCarlo.SobolSample(R=QuasiMonteCarlo.Shift(rng=rng))
    candidates = QuasiMonteCarlo.sample(ts.n_candidates, d, shifted_sobol)
    draw = rand(rng, post(ColVecs(candidates), ts.jitter))
    return candidates[:, argmin(draw)]
end
