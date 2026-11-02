using CairoMakie
using JLD2
using Statistics: median, quantile

const RESULTS_DIR = joinpath(@__DIR__, "counting_ones_results")
const OUTDIR = joinpath(@__DIR__, "..", "docs", "benchmarks")
const SAMPLER_COLORS = Dict("Random" => Makie.wong_colors()[1], "Hyperband" => Makie.wong_colors()[3], "DEHB" => Makie.wong_colors()[2])

const PLOTS = (
    (ncat=32, ncont=32, samplers=("Random", "Hyperband", "DEHB"), xlims=(0.015, 1e6), ylims=nothing),
    (ncat=8, ncont=8, samplers=("Random", "Hyperband", "DEHB"), xlims=(1e-1, 1e4), ylims=(1e-3, 1e0)),
)

results_path(ncat, ncont, name) = joinpath(RESULTS_DIR, "counting_ones_$(ncat)cat_$(ncont)cont_$(name).jld2")

columnwise(f, curves) = [any(isnan, c) || f(c) <= 0 ? NaN : f(c) for c in eachcol(curves)]

function plot_counting_ones(; ncat, ncont, samplers, xlims, ylims)
    fig = Figure(size=(750, 400))
    ax = Axis(fig[1, 1]; xlabel="cumulative budget / b_max", ylabel="normalized regret",
              xscale=log10, yscale=log10,
              xticks=Makie.LogTicks(-2:6),
              xminorticksvisible=true, xminorgridvisible=true,
              xminorticks=[d * 10.0^e for e in -2:5 for d in 2:9],
              yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(9))
    repeats = 0
    for name in samplers
        saved = JLD2.load(results_path(ncat, ncont, name))
        grid, curves = saved["grid"], saved["curves"]
        repeats = size(curves, 1)
        color = SAMPLER_COLORS[name]
        lines!(ax, grid, columnwise(median, curves); label=name, color=color)
        band!(ax, grid, columnwise(c -> quantile(c, 0.25), curves), columnwise(c -> quantile(c, 0.75), curves); color=(color, 0.15))
    end
    ax.title = "Stochastic Counting Ones, $ncat categorical + $ncont continuous ($repeats repeats)"
    xlims!(ax, xlims...)
    if ylims !== nothing
        ylims!(ax, ylims...)
    end
    axislegend(ax; position=:lb)
    mkpath(OUTDIR)
    path = joinpath(OUTDIR, "counting_ones_$(ncat + ncont)d_regret.png")
    save(path, fig)
    println("plot written to $path")
end

for plot in PLOTS
    if all(name -> isfile(results_path(plot.ncat, plot.ncont, name)), plot.samplers)
        plot_counting_ones(; plot...)
    else
        println("$(plot.ncat) + $(plot.ncont): skipped, not every sampler has results in $RESULTS_DIR")
    end
end
