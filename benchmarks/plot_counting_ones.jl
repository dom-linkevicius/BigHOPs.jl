using CairoMakie
using JLD2
using Statistics: median, quantile

const RESULTS_DIR = joinpath(@__DIR__, "counting_ones_results")
const OUTDIR = joinpath(@__DIR__, "..", "docs", "benchmarks")
const SAMPLER_COLORS = Dict("Random" => Makie.wong_colors()[1], "Hyperband" => Makie.wong_colors()[3], "DEHB" => Makie.wong_colors()[2])

const PLOTS = (
    (ncat=32, ncont=32, samplers=("Random", "Hyperband", "DEHB"), xlims=(0.015, 1e6), ylims=(0.015, 0.5)),
    (ncat=8, ncont=8, samplers=("Random", "Hyperband", "DEHB"), xlims=(1e-1, 1e4), ylims=(1e-3, 1e0)),
)

const GRID = exp10.(range(-2, 6; length=160))

results_path(ncat, ncont, name) = joinpath(RESULTS_DIR, "counting_ones_$(ncat)cat_$(ncont)cont_$(name).jld2")

best_measured_regret(trials, d) = (accumulate(min, trials.value) .+ d) ./ d

function on_grid(budget, curve)
    out, j = fill(NaN, length(GRID)), 0
    for (i, g) in enumerate(GRID)
        while j < length(budget) && budget[j+1] <= g
            j += 1
        end
        if j > 0 && g <= budget[end]
            out[i] = curve[j]
        end
    end
    return out
end

function load_curves(ncat, ncont, name; curve=best_measured_regret)
    JLD2.jldopen(results_path(ncat, ncont, name), "r") do file
        repeat_keys = sort(filter(startswith("repeat_"), collect(keys(file))); by=k -> parse(Int, k[8:end]))
        rows = map(repeat_keys) do key
            trials = file[key]
            on_grid(cumsum(Float64.(trials.r)) ./ file["r_max"], curve(trials, ncat + ncont))
        end
        return permutedims(reduce(hcat, rows))
    end
end

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
        curves = load_curves(ncat, ncont, name)
        repeats = size(curves, 1)
        color = SAMPLER_COLORS[name]
        lines!(ax, GRID, columnwise(median, curves); label=name, color=color)
        band!(ax, GRID, columnwise(c -> quantile(c, 0.25), curves), columnwise(c -> quantile(c, 0.75), curves); color=(color, 0.15))
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
