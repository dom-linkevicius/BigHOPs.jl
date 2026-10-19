"""
    Domain

Abstract type for one hyperparameter's search space: `Continuous`, `Ordinal` or `Nominal`.
"""
abstract type Domain end

"""
    Nominal(levels::Int)
    Nominal(values::AbstractVector)

An unordered set of choices, of any type, including functions. `Nominal(k)` is shorthand for the choices `1:k`.
"""
struct Nominal{V<:AbstractVector} <: Domain
    values::V
    function Nominal{V}(values::V) where {V<:AbstractVector}
        isempty(values) && throw(ArgumentError("values must be non-empty"))
        return new{V}(values)
    end
end
Nominal(values::AbstractVector) = Nominal{typeof(values)}(values)
Nominal(levels::Int) = Nominal(Base.OneTo(levels))

"""
    Ordinal(levels::Int)
    Ordinal(values::AbstractVector)

An ordered set of choices. Numeric values must be sorted ascending; non-numeric values are taken in the order given, with a warning. `Ordinal(k)` is shorthand for the choices `1:k`.
"""
struct Ordinal{V<:AbstractVector} <: Domain
    values::V
    function Ordinal{V}(values::V) where {V<:AbstractVector}
        isempty(values) && throw(ArgumentError("values must be non-empty"))
        _check_order(values)
        return new{V}(values)
    end
end
Ordinal(values::AbstractVector) = Ordinal{typeof(values)}(values)
Ordinal(levels::Int) = Ordinal(Base.OneTo(levels))

_check_order(values::AbstractVector{<:Real}) =
    issorted(values) || throw(ArgumentError("Ordinal requires numeric values to be sorted in increasing order; got $values -- use Nominal if order doesn't apply"))
_check_order(values::AbstractVector) =
    @warn "Ordinal cannot verify order for non-numeric values (default isless doesn't reliably match intended domain order); assuming this is the intended order" values

"""
    Continuous(min, max; transform=identity)

A real interval `[min, max]`, passed through `transform` after sampling, so `Continuous(-4, -1; transform=exp10)` is log-uniform over `[1e-4, 1e-1]`. `transform` must be finite at both ends, and should be a named function rather than a closure if the optimizer will be checkpointed.
"""
struct Continuous{F} <: Domain
    min::Float64
    max::Float64
    transform::F
    function Continuous{F}(lo::Float64, hi::Float64, transform::F) where {F}
        hi > lo || throw(ArgumentError("max ($hi) must be greater than min ($lo)"))
        (isfinite(transform(lo)) && isfinite(transform(hi))) ||
            throw(ArgumentError("transform must be finite at both ends of [$lo, $hi]; got non-finite values"))
        return new{F}(lo, hi, transform)
    end
end
Continuous(min::Real, max::Real; transform=identity) =
    Continuous{typeof(transform)}(Float64(min), Float64(max), transform)

"""
    length(d::Union{Nominal,Ordinal})

Number of choices in `d`.
"""
Base.length(d::Union{Nominal,Ordinal}) = length(d.values)

"""
    from_unit(d::Domain, u::Real)

Decode a sampler's unit value `u ∈ [0, 1]` into a value of `d`. Discrete domains split `[0, 1]` into equal bins in order; `Continuous` maps linearly onto `[min, max]` and applies `transform`.
"""
from_unit(d::Union{Nominal,Ordinal}, u::Real) = d.values[clamp(ceil(Int, u * length(d)), 1, length(d))]
from_unit(d::Continuous, u::Real) = d.transform(d.min + clamp(u, 0.0, 1.0) * (d.max - d.min))
