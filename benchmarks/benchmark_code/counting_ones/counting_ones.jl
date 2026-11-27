using BigHOPs
using Distributions: Binomial
using StableRNGs: StableRNG

include(joinpath(@__DIR__, "..", "gp_sampler.jl"))

const R_MIN = 9
const R_MAX = 729
const ETA = 3

function counting_ones_candidates(ncat::Int, ncont::Int)
    cat_keys = Tuple(Symbol("cat$i") for i in 1:ncat)
    cont_keys = Tuple(Symbol("cont$j") for j in 1:ncont)
    return NamedTuple{(cat_keys..., cont_keys...)}((ntuple(_ -> Nominal([0, 1]), ncat)..., ntuple(_ -> Continuous(0, 1), ncont)...))
end

true_regret(p, d::Int) = 1 - sum(values(p)[(end-d+1):end]) / d

function noisy_f(p, b::Int, ncat::Int, ncont::Int)
    v = values(p)
    cont = 0.0
    for x in v[(end-ncont+1):end]
        cont += rand(Binomial(b, x)) / b
    end
    return -(sum(v[(end-ncat-ncont+1):(end-ncont)]) + cont)
end

function _iteration_budget()
    smax = BigHOPs._smax(R_MAX, R_MIN, ETA)
    return sum(BigHOPs._capacity(R_MAX, R_MIN, ETA, bracket, rung) * BigHOPs._resource(R_MAX, R_MIN, ETA, bracket, rung)
               for bracket in 1:(smax+1) for rung in 1:BigHOPs._n_rungs(R_MAX, R_MIN, ETA, bracket))
end
const ITERATION_BUDGET = _iteration_budget()
_iterations(target) = max(1, ceil(Int, target / (ITERATION_BUDGET / R_MAX)))

function _make(name::String, target::Float64, seed::Int, ncat::Int, ncont::Int)
    candidates = counting_ones_candidates(ncat, ncont)
    full(p) = noisy_f(p, R_MAX, ncat, ncont)
    sched(p; pre_artefact=nothing) = (noisy_f(p, p.r, ncat, ncont), nothing)
    plain(p) = noisy_f(p, p.r, ncat, ncont)
    name == "Random" &&
        return Hyperoptimizer(full, candidates; sampler=RandomSampler(StableRNG(seed)), n=round(Int, target))
    name == "GP" &&
        return Hyperoptimizer(full, candidates;
                              sampler=gp_sampler(StableRNG(seed)), n=round(Int, target))
    name == "Hyperband" &&
        return Hyperoptimizer(Stateful(sched), candidates,
                              Hyperband(R=R_MAX, η=ETA, r_min=R_MIN, iterations=_iterations(target),
                                        inner=RandomSampler(StableRNG(seed))))
    name == "BOHB" &&
        return Hyperoptimizer(Stateful(sched), candidates,
                              BOHB(R=R_MAX, η=ETA, r_min=R_MIN, iterations=_iterations(target), rng=StableRNG(seed)))
    return Hyperoptimizer(plain, candidates,
                          DEHB(R=R_MAX, η=ETA, r_min=R_MIN, iterations=_iterations(target), rng=StableRNG(seed)))
end

function _trials(ho, d::Int)
    told = filter(e -> !ismissing(e.value), ho.runs)
    return (r=UInt16[get(e.params, :r, R_MAX) for e in told], value=Float32[e.value for e in told],
            true_regret=Float32[true_regret(e.params, d) for e in told])
end

function _run_repeat(name::String, target::Float64, repeat::Int, ncat::Int, ncont::Int)
    ho = _make(name, target, repeat, ncat, ncont)
    elapsed = @elapsed run!(ho; executor=Serial())
    return _trials(ho, ncat + ncont), elapsed
end

function run_task(task)
    (; sampler, target, repeat, ncat, ncont) = task
    trials, elapsed = _run_repeat(sampler, target, repeat, ncat, ncont)
    println("$sampler repeat=$repeat  $(round(elapsed; digits=1))s  trials=$(length(trials.r))  " *
            "budget/b_max=$(round(sum(Float64, trials.r) / R_MAX; digits=1))  " *
            "best measured regret=$(round((minimum(trials.value) + ncat + ncont) / (ncat + ncont); digits=5))")
    flush(stdout)
    return (; ncat, ncont, target, r_max=R_MAX, trials)
end
