"""
    LHSampler(; rng=StableRNG(1))

Draw all `n` trials up front as a Latin hypercube, which spreads them evenly along every hyperparameter. Warns if `n` is too small to cover every combination of discrete values. Each `LHSampler()` can back only one `Hyperoptimizer`.
"""
struct LHSampler{T<:Random.AbstractRNG} <: Sampler
    rng::T
    design::Matrix{Float64}
end

LHSampler(; rng::Random.AbstractRNG=StableRNG(1)) = LHSampler(rng, Matrix{Float64}(undef, 0, 0))

_discrete_product(candidates) = prod((length(d) for d in candidates if d isa Union{Nominal,Ordinal}); init=1)

function init(s::LHSampler, candidates, n)
    isempty(s.design) ||
        throw(ArgumentError("LHSampler: this sampler is already initialized with a design for $(size(s.design, 2)) trials -- pass a fresh LHSampler(), since re-initializing would silently replace that design with one drawn for different candidates or a different n"))
    product = _discrete_product(candidates)
    n < product && @warn "LHSampler: n ($n) is less than the number of discrete-variable combinations ($product) -- not every combination can be covered with this budget"
    design = QuasiMonteCarlo.sample(n, length(candidates), QuasiMonteCarlo.LatinHypercubeSample(rng=s.rng))
    n >= product && _warn_missing_combinations(design, candidates)
    return LHSampler(s.rng, design)
end

function _warn_missing_combinations(design::Matrix{Float64}, candidates)
    discrete_dims = findall(d -> d isa Union{Nominal,Ordinal}, candidates)
    isempty(discrete_dims) && return nothing
    covered = Set(Tuple(from_unit(candidates[dim], design[dim, col]) for dim in discrete_dims) for col in axes(design, 2))
    all_combos = vec(collect(Iterators.product((candidates[dim].values for dim in discrete_dims)...)))
    missing_combos = filter(c -> c ∉ covered, all_combos)
    isempty(missing_combos) && return nothing
    suffix = length(missing_combos) > 20 ? " (and $(length(missing_combos) - 20) more)" : ""
    @warn "LHSampler: the design doesn't cover every discrete-variable combination$suffix" uncovered = first(missing_combos, 20)
    return nothing
end

(s::LHSampler)(candidates, runs) = s.design[:, length(runs)+1]

on_tell!(::LHSampler, runs, entry) = nothing
exhausted(s::LHSampler, ho) = length(ho.runs) >= size(s.design, 2)
blocked(::LHSampler, ho) = false
create_run_entry(::LHSampler, ho, id, params, unit_params) = RunEntry(id, params, unit_params)
