using BigHOPs
using CairoMakie
using Distributions: Binomial
using StableRNGs: StableRNG
using Statistics: median, quantile
using JLD2

const R_MIN = 9
const R_MAX = 729
const ETA = 3

const OUTDIR = joinpath(@__DIR__, "..", "docs", "benchmarks")

function counting_ones_candidates(ncat::Int, ncont::Int)
    cat_keys = Tuple(Symbol("cat$i") for i in 1:ncat)
    cont_keys = Tuple(Symbol("cont$j") for j in 1:ncont)
    return NamedTuple{(cat_keys..., cont_keys...)}((ntuple(_ -> Nominal([0, 1]), ncat)..., ntuple(_ -> Continuous(0, 1), ncont)...))
end

true_f(p, d::Int) = -sum(values(p)[(end-d+1):end])

regret(p, d::Int) = (true_f(p, d) + d) / d

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

const SAMPLER_COLORS = Dict("Random" => Makie.wong_colors()[1], "Hyperband" => Makie.wong_colors()[3], "DEHB" => Makie.wong_colors()[2])

function _make(name::String, target::Float64, seed::Int, ncat::Int, ncont::Int)
    candidates = counting_ones_candidates(ncat, ncont)
    full(p) = noisy_f(p, R_MAX, ncat, ncont)
    sched(p; pre_artefact=nothing) = (noisy_f(p, p.r, ncat, ncont), nothing)
    plain(p) = noisy_f(p, p.r, ncat, ncont)
    name == "Random" &&
        return Hyperoptimizer(full, candidates; sampler=RandomSampler(StableRNG(seed)), n=round(Int, target))
    name == "Hyperband" &&
        return Hyperoptimizer(Stateful(sched), candidates,
                              Hyperband(R=R_MAX, η=ETA, r_min=R_MIN, iterations=_iterations(target),
                                        inner=RandomSampler(StableRNG(seed))))
    return Hyperoptimizer(plain, candidates,
                          DEHB(R=R_MAX, η=ETA, r_min=R_MIN, iterations=_iterations(target), rng=StableRNG(seed)))
end

function _curve(ho, d::Int)
    spent, best, cum, seen, incumbent = Float64[], Float64[], 0, Inf, 1.0
    for e in ho.runs
        ismissing(e.value) && continue
        cum += get(e.params, :r, R_MAX)
        if e.value < seen
            seen = e.value
            incumbent = regret(e.params, d)
        end
        push!(spent, cum / R_MAX)
        push!(best, incumbent)
    end
    return spent, best
end

function curves_for(name::String, target::Float64; ncat::Int, ncont::Int, repeats::Int, cachedir::String)
    path = joinpath(cachedir, "counting_ones_$(ncat + ncont)d_$(name)_$(round(Int, target)).jld2")
    isfile(path) && return JLD2.load(path, "curves")
    curves = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
    for repeat in 1:repeats
        ho = _make(name, target, repeat, ncat, ncont)
        elapsed = @elapsed run!(ho; executor=Serial(), show_progress=false)
        spent, best = _curve(ho, ncat + ncont)
        push!(curves, (spent, best))
        println("$name repeat=$repeat  $(round(elapsed; digits=1))s  trials=$(length(ho.runs))  " *
                "budget/b_max=$(round(spent[end]; digits=1))  final regret=$(round(best[end]; digits=5))")
        flush(stdout)
        ho = nothing
        GC.gc()
    end
    mkpath(cachedir)
    JLD2.save(path, "curves", curves)
    return curves
end

const GRID = exp10.(range(-2, 6; length=160))

function _on_grid(spent, best)
    out, j = fill(NaN, length(GRID)), 0
    for (i, g) in enumerate(GRID)
        while j < length(spent) && spent[j+1] <= g
            j += 1
        end
        if j > 0 && g <= spent[end]
            out[i] = best[j]
        end
    end
    return out
end

function plot_counting_ones(; ncat::Int, ncont::Int, samplers::NamedTuple, repeats::Int, cachedir::String)
    fig = Figure(size=(750, 400))
    ax = Axis(fig[1, 1]; xlabel="cumulative budget / b_max", ylabel="normalized regret",
              xscale=log10, yscale=log10,
              xticks=Makie.LogTicks(-2:6),
              xminorticksvisible=true, xminorgridvisible=true,
              xminorticks=[d * 10.0^e for e in -2:5 for d in 2:9],
              yticks=([0.1], [rich("10", superscript("-1"))]),
              yminorticksvisible=true, yminorgridvisible=true,
              yminorticks=vcat(0.01:0.01:0.09, 0.1:0.1:0.4),
              title="Stochastic Counting Ones, $ncat categorical + $ncont continuous ($repeats repeats)")
    for (name, target) in pairs(samplers)
        curves = curves_for(String(name), Float64(target); ncat, ncont, repeats, cachedir)
        color = SAMPLER_COLORS[String(name)]
        m = reduce(vcat, (_on_grid(s, b)' for (s, b) in curves))
        med = [any(isnan, c) ? NaN : median(c) for c in eachcol(m)]
        lo = [any(isnan, c) ? NaN : quantile(c, 0.25) for c in eachcol(m)]
        hi = [any(isnan, c) ? NaN : quantile(c, 0.75) for c in eachcol(m)]
        lines!(ax, GRID, med; label=String(name), color=color)
        band!(ax, GRID, lo, hi; color=(color, 0.15))
    end
    xlims!(ax, 0.015, 1e6)
    ylims!(ax, 0.015, 0.5)
    axislegend(ax; position=:lb)
    mkpath(OUTDIR)
    save(joinpath(OUTDIR, "counting_ones_$(ncat + ncont)d_regret.png"), fig)
end
