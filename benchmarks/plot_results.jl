using CairoMakie
using JLD2
using Statistics: median, quantile

const RESULTS_PATH = joinpath(@__DIR__, "results.jld2")
const OUTDIR = joinpath(@__DIR__, "..", "docs", "benchmarks")
mkpath(OUTDIR)

const REGRET_GRID_POINTS = 40
const SAMPLER_NAMES = ("Random", "LHS", "Hyperband", "ASHA", "DEHB", "GP")
const SAMPLER_COLORS = Dict(zip(SAMPLER_NAMES, Makie.wong_colors()))
const SHA_SAMPLER_NAMES = ("Hyperband", "ASHA")
const Y_UPPER_LIMIT = 0.11
const Y_TICK_EXPONENTS = [-3, -2, -1]
const X_LOWER_LIMIT = 1e-1
const X_UPPER_LIMIT = 600

"""
    _resample_to_grid(times, running_min, grid)
"""
function _resample_to_grid(times, running_min, grid)
    out = fill(NaN, length(grid))
    j = 0
    for (i, g) in enumerate(grid)
        while j < length(times) && times[j+1] <= g
            j += 1
        end
        j > 0 && (out[i] = running_min[j])
    end
    return out
end

function plot_results(runs, metadata)
    regret_repeats = metadata.regret_repeats
    all_values = Float64[e for curves in values(runs) for (_, best) in curves for e in best]
    global_best = minimum(all_values)

    computed = Dict{Tuple{String,Symbol},NamedTuple}()
    ymin = Inf
    for name in SAMPLER_NAMES, ex_name in (:Serial, :Threaded)
        curves = runs[(name, ex_name)]
        own_min = minimum(times[1] for (times, _) in curves if !isempty(times))
        own_max = maximum(times[end] for (times, _) in curves if !isempty(times))
        grid = exp10.(range(log10(own_min), log10(own_max); length=REGRET_GRID_POINTS))
        curves_on_grid = Matrix{Float64}(undef, length(curves), length(grid))
        for (r, (times, best)) in enumerate(curves)
            curves_on_grid[r, :] = _resample_to_grid(times, best, grid)
        end
        regret = [any(isnan, col) ? NaN : max(median(col) - global_best, 1e-4) for col in eachcol(curves_on_grid)]
        lower = [any(isnan, col) ? NaN : max(quantile(col, 0.25) - global_best, 1e-4) for col in eachcol(curves_on_grid)]
        upper = [any(isnan, col) ? NaN : max(quantile(col, 0.75) - global_best, 1e-4) for col in eachcol(curves_on_grid)]
        computed[(name, ex_name)] = (; grid, regret, lower, upper)
        finite_lower = filter(!isnan, lower)
        isempty(finite_lower) || (ymin = min(ymin, minimum(finite_lower)))
    end

    fig = Figure(size=(750, 500))
    legend_source = nothing
    for (row, ex_name, title_word) in ((1, :Serial, "Serial"), (2, :Threaded, "Threaded"))
        ax = Axis(fig[row, 1]; xlabel=(row == 2 ? "wall-clock time (s)" : ""),
                  xscale=log10, yscale=log10, yticks=Makie.LogTicks(Y_TICK_EXPONENTS),
                  xminorticksvisible=true, xminorgridvisible=true, xminorticks=IntervalsBetween(9),
                  yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(9),
                  title="Wall-clock regret comparison -- $title_word ($regret_repeats repeats/combo, $(metadata.nthreads) threads)")
        for name in SAMPLER_NAMES
            c = computed[(name, ex_name)]
            color = SAMPLER_COLORS[name]
            label_name = name in SHA_SAMPLER_NAMES ? "$name/$(metadata.sha_inner_sampler)" : name
            lines!(ax, c.grid, c.regret; label=label_name, color=color)
            band!(ax, c.grid, c.lower, c.upper; color=(color, 0.15))
        end
        xlims!(ax, X_LOWER_LIMIT, X_UPPER_LIMIT)
        ylims!(ax, ymin, Y_UPPER_LIMIT)
        legend_source = ax
    end
    Label(fig[1:2, 0], "regret (best validation loss - global best)"; rotation=pi / 2, tellheight=false)
    Legend(fig[0, 1], legend_source; orientation=:horizontal, nbanks=1)
    save(joinpath(OUTDIR, "wallclock_regret_comparison.png"), fig)
end

@info "Loading results from $RESULTS_PATH..."
runs, metadata = JLD2.load(RESULTS_PATH, "runs", "metadata")
plot_results(runs, metadata)
@info "Plot written to $OUTDIR"
