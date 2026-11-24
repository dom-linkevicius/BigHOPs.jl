using CairoMakie
using JLD2
using Statistics: median, quantile

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "nn")

results_dir(name, ex_name) = joinpath(RESULTS_DIR, "$(name)_$(ex_name)")

const OUTDIR = joinpath(@__DIR__, "..", "..", "..", "docs", "src", "figures")
mkpath(OUTDIR)

const REGRET_GRID_POINTS = 40
const SAMPLER_NAMES = ("Random", "Hyperband", "ASHA", "DEHB", "BOHB")
const SAMPLER_COLORS = Dict(zip(SAMPLER_NAMES, Makie.wong_colors()))
const Y_UPPER_LIMIT = 0.07
const Y_TICKS = 0.02 .* sqrt(2) .^ (-1:3)
const X_LOWER_LIMIT = 1e0
const X_UPPER_LIMIT = 2e3

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

function _median_ci(col)
    n = length(col)
    half_width = 1.96 * sqrt(n) / 2
    sorted = sort(col)
    return sorted[max(floor(Int, n / 2 - half_width), 1)], sorted[min(ceil(Int, n / 2 + 1 + half_width), n)]
end

function plot_results(runs, metadata)
    fewest, most = extrema(length, values(runs))
    regret_repeats = fewest == most ? fewest : "$(fewest)–$(most)"
    all_values = Float64[e for curves in values(runs) for (_, best) in curves for e in best]
    global_best = minimum(all_values)

    computed = Dict{Tuple{String,Symbol},NamedTuple}()
    ymin = Inf
    for name in SAMPLER_NAMES, ex_name in (:Serial, :Threaded)
        haskey(runs, (name, ex_name)) || continue
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
        bounds = [any(isnan, col) ? (NaN, NaN) : max.(_median_ci(col) .- global_best, 1e-4) for col in eachcol(curves_on_grid)]
        computed[(name, ex_name)] = (; grid, regret, ci_lower=first.(bounds), ci_upper=last.(bounds))
        finite_lower = filter(!isnan, lower)
        isempty(finite_lower) || (ymin = min(ymin, minimum(finite_lower)))
    end

    fig = Figure(size=(750, 600), fontsize=18)
    legend_source = nothing
    for (row, ex_name, title_word) in ((1, :Serial, "Serial"), (2, :Threaded, "Threaded"))
        ax = Axis(fig[row, 1]; xlabel=(row == 2 ? "wall-clock time (s)" : ""),
                  xscale=log10, yscale=log10, yticks=Y_TICKS, ytickformat="{:.3f}",
                  xminorticksvisible=true, xminorgridvisible=true, xminorticks=IntervalsBetween(9),
                  yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(2),
                  title=(row == 2 ? "$title_word ($(metadata.nthreads) threads)" : title_word))
        for name in SAMPLER_NAMES
            haskey(computed, (name, ex_name)) || continue
            c = computed[(name, ex_name)]
            color = SAMPLER_COLORS[name]
            defined = .!isnan.(c.regret)
            band!(ax, c.grid[defined], c.ci_lower[defined], c.ci_upper[defined]; color=(color, 0.15))
            lines!(ax, c.grid, c.regret; label=name, color=color)
        end
        xlims!(ax, X_LOWER_LIMIT, X_UPPER_LIMIT)
        ylims!(ax, ymin, Y_UPPER_LIMIT)
        legend_source = ax
    end
    Label(fig[1:2, 0], "regret (best validation loss - global best)"; rotation=pi / 2, tellheight=false)
    Legend(fig[0, 1], legend_source; orientation=:horizontal)
    Label(fig[-1, 1], "Wall-clock regret comparison ($regret_repeats repeats/combo)"; font=:bold, fontsize=20, tellwidth=false)
    save(joinpath(OUTDIR, "wallclock_regret_comparison.png"), fig)
end

function load_runs()
    runs = Dict{Tuple{String,Symbol},Vector{Tuple{Vector{Float64},Vector{Float64}}}}()
    metadata = nothing
    for name in SAMPLER_NAMES, ex_name in (:Serial, :Threaded)
        isdir(results_dir(name, ex_name)) || continue
        paths = filter(endswith(".jld2"), readdir(results_dir(name, ex_name); join=true))
        isempty(paths) && continue
        metadata = JLD2.load(first(paths), "metadata")
        runs[(name, ex_name)] = [JLD2.load(path, "curve") for path in paths]
    end
    return runs, metadata
end

@info "Loading results from $RESULTS_DIR..."
runs, metadata = load_runs()
plot_results(runs, metadata)
@info "Plot written to $OUTDIR"
