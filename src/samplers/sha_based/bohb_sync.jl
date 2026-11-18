struct BOHBModel
    r::Int
    l::MultivariateDistribution
    g::MultivariateDistribution
    l′::MultivariateDistribution
end

"""
    BOHBSampler(; ρ=1/3, q=0.15, N_s=64, N_min=nothing, b_w=3, min_bandwidth=1e-3, rng=StableRNG(1))

The kernel-density sampler inside `BOHB`; construct `BOHB` rather than using this directly.
"""
struct BOHBSampler{T<:Random.AbstractRNG} <: Sampler
    ρ::Float64
    q::Float64
    N_s::Int
    N_min::Union{Nothing,Int}
    b_w::Float64
    min_bandwidth::Float64
    rng::T
    observations::Dict{Int,Vector{Int}}
    model::Base.RefValue{Union{Nothing,BOHBModel}}
    stale::Base.RefValue{Bool}
end
function BOHBSampler(; ρ::Real=1 / 3, q::Real=0.15, N_s::Int=64, N_min::Union{Nothing,Int}=nothing, b_w::Real=3,
                     min_bandwidth::Real=1e-3, rng::Random.AbstractRNG=StableRNG(1))
    0 <= ρ <= 1 || throw(ArgumentError("ρ must be in [0, 1], got $ρ"))
    0 < q < 1 || throw(ArgumentError("q must be in (0, 1), got $q"))
    N_s > 0 || throw(ArgumentError("N_s must be positive, got $N_s"))
    N_min === nothing || N_min > 0 || throw(ArgumentError("N_min must be positive, got $N_min"))
    b_w > 0 || throw(ArgumentError("b_w must be positive, got $b_w"))
    min_bandwidth > 0 || throw(ArgumentError("min_bandwidth must be positive, got $min_bandwidth"))
    return BOHBSampler(Float64(ρ), Float64(q), N_s, N_min, Float64(b_w), Float64(min_bandwidth), rng,
                       Dict{Int,Vector{Int}}(), Ref{Union{Nothing,BOHBModel}}(nothing), Ref(false))
end

init(s::BOHBSampler, candidates, n) = s

_bohb_standalone() = throw(ArgumentError("BOHBSampler can't be used on its own; construct BOHB(; R, ...) instead"))
on_tell!(::BOHBSampler, runs, entry) = _bohb_standalone()
exhausted(::BOHBSampler, ho) = _bohb_standalone()
blocked(::BOHBSampler, ho) = _bohb_standalone()
create_run_entry(::BOHBSampler, ho, id, params, unit_params) = _bohb_standalone()
(::BOHBSampler)(candidates, runs) = _bohb_standalone()

"""
    BOHB(; R, η=3, r_min=1, iterations=1, ρ=1/3, q=0.15, N_s=64, N_min=nothing, b_w=3,
         min_bandwidth=1e-3, rng=StableRNG(1))

`Hyperband`'s schedule with the first rung of each bracket drawn from kernel density
estimates, following the paper (Falkner et al., 2018, https://arxiv.org/abs/1807.01774),
with the details it leaves open taken from the reference implementation
(https://github.com/automl/HpBandSter). Promotions are as in `Hyperband`.

A fraction `ρ` of draws, and every draw before a resource level has enough completed
trials, is uniform at random. Otherwise the completed trials at the largest such level are
split into the best `q` and the rest, and the draw is whichever of `N_s` samples from the
good density, with bandwidths widened by `b_w`, has the highest ratio of good to bad
density. `N_min` defaults to the number of hyperparameters plus one.
"""
const BOHB = SuccessiveHalving{true,<:BOHBSampler}

SuccessiveHalving{true,<:BOHBSampler}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1, ρ::Real=1 / 3,
                                       q::Real=0.15, N_s::Int=64, N_min::Union{Nothing,Int}=nothing, b_w::Real=3,
                                       min_bandwidth::Real=1e-3, rng::Random.AbstractRNG=StableRNG(1)) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, iterations=iterations,
                            inner=BOHBSampler(; ρ=ρ, q=q, N_s=N_s, N_min=N_min, b_w=b_w,
                                              min_bandwidth=min_bandwidth, rng=rng))

function _init_inner(inner::BOHBSampler, ::SuccessiveHalving, candidates)
    d = length(candidates)
    N_min = something(inner.N_min, d + 1)
    # Paper leaves N_min free (d + 1 in its experiments); HpBandSter requires at least d + 1.
    N_min > d || throw(ArgumentError("N_min must be at least the number of hyperparameters plus one ($(d + 1)), got $N_min"))
    return BOHBSampler(inner.ρ, inner.q, inner.N_s, N_min, inner.b_w, inner.min_bandwidth, inner.rng,
                       Dict{Int,Vector{Int}}(), Ref{Union{Nothing,BOHBModel}}(nothing), Ref(false))
end

function _tell_inner!(b::BOHBSampler, s, runs, entry)
    entry.status === Completed || return nothing
    r = entry.params.r
    push!(get!(b.observations, r, Int[]), entry.id)
    if b.model[] === nothing || r >= b.model[].r
        b.stale[] = true
    end
    return nothing
end

# Paper fits from N_min + 2 trials; HpBandSter needs more than d in each set.
_fittable(N_min::Int, q::Real, d::Int, N::Int) = N - max(N_min, floor(Int, q * N)) > d
_fittable(b::BOHBSampler, d::Int) = any(ids -> _fittable(b.N_min, b.q, d, length(ids)), values(b.observations))

_kde_value(::Domain, u) = u
_kde_value(d::Nominal, u) = clamp(ceil(Int, u * length(d)), 1, length(d))

_bandwidth_cap(::Domain) = Inf
_bandwidth_cap(d::Nominal) = (length(d) - 1) / length(d)

function _kde_data(b::BOHBSampler, candidates, runs, ids)
    X = [Float64(_kde_value(c, runs[id].unit_params[j])) for (j, c) in enumerate(candidates), id in ids]
    scott = 1.06 .* vec(std(X; dims=2, corrected=false)) .* size(X, 2)^(-1 / (4 + size(X, 1)))
    # Paper: plain Scott's rule. Floor from HpBandSter for constant columns, cap from statsmodels' AA bound.
    return X, max.(min.(scott, [_bandwidth_cap(c) for c in candidates]), b.min_bandwidth)
end

# Paper names only continuous and categorical; Ordinal is Gaussian, as integers are in HpBandSter.
_kernel(::Domain, x, h) = Normal(x, h)
function _kernel(d::Nominal, k, h)
    length(d) == 1 && return Categorical(1)
    # Declared levels, not statsmodels' levels in the data, which divides by zero at one level.
    p = fill(h / (length(d) - 1), length(d))
    p[Int(k)] = 1 - h
    return Categorical(p)
end

# Paper's l′ is untruncated; HpBandSter truncates to [0, 1], the encoded range.
_proposal_kernel(::Domain, x, h, b_w) = truncated(Normal(x, b_w * h), 0.0, 1.0)
# Paper widens every bandwidth by b_w; HpBandSter doesn't on Nominal and redraws over all levels.
function _proposal_kernel(d::Nominal, k, h, b_w)
    p = fill(h / length(d), length(d))
    p[Int(k)] += 1 - h
    return Categorical(p)
end

_mixture(kernel, candidates, X, h) =
    MixtureModel([product_distribution([kernel(c, X[j, i], h[j]) for (j, c) in enumerate(candidates)]) for i in axes(X, 2)])

function _bohb_model(b::BOHBSampler, candidates, runs)
    r = maximum(r for (r, ids) in b.observations if _fittable(b.N_min, b.q, length(candidates), length(ids)))
    ranked = sort(b.observations[r]; by=id -> runs[id].value)
    N = length(ranked)
    n_good = max(b.N_min, floor(Int, b.q * N))
    n_bad = max(b.N_min, floor(Int, (1 - b.q) * N))
    # Paper's good and worst sets overlap for small N; HpBandSter takes bad as the next n_bad, disjoint.
    good = _kde_data(b, candidates, runs, ranked[1:n_good])
    bad = _kde_data(b, candidates, runs, ranked[(n_good+1):min(N, n_good + n_bad)])
    return BOHBModel(r, _mixture(_kernel, candidates, good...), _mixture(_kernel, candidates, bad...),
                     _mixture((c, x, h) -> _proposal_kernel(c, x, h, b.b_w), candidates, good...))
end

_to_unit(::Domain, x) = x
_to_unit(d::Nominal, k) = (k - 0.5) / length(d)

function _sample_sh_inner(s::BOHB, candidates, runs, ::SHDecision{:draw})
    b = s.inner
    rand(b.rng) < b.ρ && return rand(b.rng, length(candidates))
    _fittable(b, length(candidates)) || return rand(b.rng, length(candidates))
    if b.stale[]
        b.model[] = _bohb_model(b, candidates, runs)
        b.stale[] = false
    end
    model = b.model[]
    best, best_score = Float64[], Inf
    for _ in 1:b.N_s
        x = rand(b.rng, model.l′)
        # Paper takes l/g directly; HpBandSter floors both at 1e-32 to avoid dividing by zero.
        score = max(1e-32, pdf(model.g, x)) / max(pdf(model.l, x), 1e-32)
        if score < best_score
            best, best_score = x, score
        end
    end
    return [_to_unit(c, x) for (c, x) in zip(candidates, best)]
end
