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
                       Dict{Int,Vector{Int}}())
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
    N_min > d || throw(ArgumentError("N_min must be at least the number of hyperparameters plus one ($(d + 1)), got $N_min"))
    return BOHBSampler(inner.ρ, inner.q, inner.N_s, N_min, inner.b_w, inner.min_bandwidth, inner.rng,
                       Dict{Int,Vector{Int}}())
end

function _tell_inner!(b::BOHBSampler, s, runs, entry)
    entry.status === Completed && push!(get!(b.observations, entry.params.r, Int[]), entry.id)
    return nothing
end

_bohb_levels(candidates) = [d isa Nominal ? length(d) : 0 for d in candidates]

function _bohb_kde(b::BOHBSampler, levels, runs, ids)
    data = [Float64(c > 0 ? clamp(ceil(Int, runs[id].unit_params[j] * c), 1, c) : runs[id].unit_params[j])
            for id in ids, (j, c) in enumerate(levels)]
    n, d = size(data)
    scott = 1.06 .* vec(std(data; dims=1, corrected=false)) .* n^(-1 / (4 + d))
    cap = [c > 0 ? (c - 1) / c : Inf for c in levels]
    return (data=data, bw=max.(min.(scott, cap), b.min_bandwidth))
end

function _bohb_model(b::BOHBSampler, levels, runs)
    d = length(levels)
    for r in sort!(collect(keys(b.observations)); rev=true)
        ranked = sort(b.observations[r]; by=id -> runs[id].value)
        N = length(ranked)
        n_good = min(N, max(b.N_min, floor(Int, b.q * N)))
        n_bad = max(b.N_min, floor(Int, (1 - b.q) * N))
        good, bad = ranked[1:n_good], ranked[(n_good+1):min(N, n_good + n_bad)]
        length(good) > d && length(bad) > d &&
            return (good=_bohb_kde(b, levels, runs, good), bad=_bohb_kde(b, levels, runs, bad))
    end
    return nothing
end

function _bohb_pdf(kde, levels, x)
    total = 0.0
    for i in axes(kde.data, 1)
        k = 1.0
        for (j, c) in enumerate(levels)
            if c > 0
                k *= kde.data[i, j] == x[j] ? 1 - kde.bw[j] : kde.bw[j] / (c - 1)
            else
                k *= pdf(Normal(kde.data[i, j], kde.bw[j]), x[j])
            end
        end
        total += k
    end
    return total / size(kde.data, 1)
end

function _bohb_sample(b::BOHBSampler, kde, levels)
    i = rand(b.rng, axes(kde.data, 1))
    x = zeros(length(levels))
    for (j, c) in enumerate(levels)
        if c == 0
            x[j] = rand(b.rng, truncated(Normal(kde.data[i, j], b.b_w * kde.bw[j]), 0.0, 1.0))
        elseif rand(b.rng) < 1 - kde.bw[j]
            x[j] = kde.data[i, j]
        else
            x[j] = rand(b.rng, 1:c)
        end
    end
    return x
end

function _sample_sh_inner(s::BOHB, candidates, runs, ::SHDecision{:draw})
    b = s.inner
    levels = _bohb_levels(candidates)
    model = _bohb_model(b, levels, runs)
    (model === nothing || rand(b.rng) < b.ρ) && return rand(b.rng, length(candidates))
    best, best_score = zeros(length(levels)), Inf
    for _ in 1:b.N_s
        x = _bohb_sample(b, model.good, levels)
        score = max(1e-32, _bohb_pdf(model.bad, levels, x)) / max(_bohb_pdf(model.good, levels, x), 1e-32)
        if score < best_score
            best, best_score = x, score
        end
    end
    return [c > 0 ? (best[j] - 0.5) / c : best[j] for (j, c) in enumerate(levels)]
end
