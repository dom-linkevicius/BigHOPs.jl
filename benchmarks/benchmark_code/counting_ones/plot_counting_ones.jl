using CairoMakie
using JLD2
using Statistics: mean, std

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "counting_ones")
const OUTDIR = joinpath(@__DIR__, "..", "..", "..", "docs", "src", "figures")
const SAMPLER_COLORS = Dict("Random" => Makie.wong_colors()[1], "Hyperband" => Makie.wong_colors()[3], "DEHB" => Makie.wong_colors()[2],
                            "BOHB" => Makie.wong_colors()[4])

const PLOTS = (
    (ncat=32, ncont=32, samplers=("Random", "Hyperband", "DEHB", "BOHB"), xlims=(0.015, 1e6), ylims=(0.015, 0.5)),
    (ncat=8, ncont=8, samplers=("Random", "Hyperband", "DEHB", "BOHB"), xlims=(1e-1, 1e4), ylims=(1e-5, 1e0)),
)

const GRID = exp10.(range(-2, 6; length=160))

results_dir(ncat, ncont, name) = joinpath(RESULTS_DIR, "counting_ones_$(ncat)cat_$(ncont)cont_$(name)")

best_measured_regret(trials, d) = (accumulate(min, trials.value) .+ d) ./ d

function incumbent_true_regret(trials, d)
    curve = Vector{Float64}(undef, length(trials.r))
    incumbent_value, incumbent_r, incumbent_regret = Inf, 0, NaN
    for i in eachindex(trials.r)
        r = Int(trials.r[i])
        if r >= incumbent_r && (r > incumbent_r || trials.value[i] < incumbent_value)
            incumbent_value, incumbent_r, incumbent_regret = trials.value[i], r, trials.true_regret[i]
        end
        curve[i] = incumbent_regret
    end
    return curve
end

function hpbandster_measured_regret(trials, d)
    curve = Vector{Float64}(undef, length(trials.r))
    incumbent_value, incumbent_r = Inf, 0
    for i in eachindex(trials.r)
        r = Int(trials.r[i])
        if r >= incumbent_r && trials.value[i] < incumbent_value
            incumbent_value, incumbent_r = trials.value[i], r
        end
        curve[i] = (incumbent_value + d) / d
    end
    return curve
end

const METRICS = (
    (title="Falkner et al. (2018)", ylabel="immediate regret",
     curves=Dict("Random" => incumbent_true_regret, "Hyperband" => incumbent_true_regret,
                 "DEHB" => incumbent_true_regret, "BOHB" => incumbent_true_regret)),
    (title="Awad et al. (2021)", ylabel="validation regret",
     curves=Dict("Random" => hpbandster_measured_regret, "Hyperband" => hpbandster_measured_regret,
                 "DEHB" => hpbandster_measured_regret, "BOHB" => hpbandster_measured_regret)),
)

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

function load_curves(ncat, ncont, name; curve=incumbent_true_regret)
    paths = filter(endswith(".jld2"), readdir(results_dir(ncat, ncont, name); join=true))
    rows = map(paths) do path
        trials, r_max = JLD2.load(path, "trials", "r_max")
        on_grid(cumsum(Float64.(trials.r)) ./ r_max, curve(trials, ncat + ncont))
    end
    return permutedims(reduce(hcat, rows))
end

columnwise(f, curves, bottom) = [any(isnan, c) ? NaN : max(f(c), bottom) for c in eachcol(curves)]

standard_error(c) = std(c) / sqrt(length(c))

function plot_counting_ones(; ncat, ncont, samplers, xlims, ylims)
    decades = ylims === nothing ? nothing : floor(Int, log10(ylims[1])):ceil(Int, log10(ylims[2]))
    bottom = ylims === nothing ? floatmin(Float64) : ylims[1] / 10
    fig = Figure(size=(1000, 450), fontsize=24, figure_padding=(16, 40, 16, 16))
    repeats = 0
    for (column, metric) in enumerate(METRICS)
        ax = Axis(fig[3, column]; title=metric.title, xlabel="cumulative budget / b_max", ylabel=metric.ylabel,
                  xscale=log10, yscale=log10,
                  xticks=Makie.LogTicks(-2:6),
                  xminorticksvisible=true, xminorgridvisible=true,
                  xminorticks=[d * 10.0^e for e in -2:5 for d in 2:9],
                  yticks=decades === nothing ? Makie.automatic : Makie.LogTicks(decades), yminorticksvisible=true, yminorgridvisible=true,
                  yminorticks=decades === nothing ? IntervalsBetween(9) : [m * 10.0^e for e in decades for m in 2:9])
        for name in samplers
            curves = load_curves(ncat, ncont, name; curve=metric.curves[name])
            repeats = size(curves, 1)
            color = SAMPLER_COLORS[name]
            lines!(ax, GRID, columnwise(mean, curves, bottom); label=name, color=color)
            band!(ax, GRID, columnwise(c -> mean(c) - standard_error(c), curves, bottom), columnwise(c -> mean(c) + standard_error(c), curves, bottom); color=(color, 0.15))
        end
        xlims!(ax, xlims...)
        if ylims !== nothing
            ylims!(ax, ylims...)
        end
        if column == 1
            Legend(fig[2, 1:length(METRICS)], ax; orientation=:horizontal, framevisible=false)
        end
    end
    Label(fig[1, :], "Stochastic Counting Ones, $ncat categorical + $ncont continuous ($repeats repeats)"; font=:bold, fontsize=24)
    mkpath(OUTDIR)
    path = joinpath(OUTDIR, "counting_ones_$(ncat + ncont)d_regret.png")
    save(path, fig)
    println("plot written to $path")
end

for plot in PLOTS
    if all(name -> isdir(results_dir(plot.ncat, plot.ncont, name)), plot.samplers)
        plot_counting_ones(; plot...)
    else
        println("$(plot.ncat) + $(plot.ncont): skipped, not every sampler has results in $RESULTS_DIR")
    end
end
