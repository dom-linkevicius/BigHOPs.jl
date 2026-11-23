module BigHOPsMakieExt

using BigHOPs
using CairoMakie
import AlgebraOfGraphics
using AlgebraOfGraphics: data, mapping, visual, draw!

# Panels are small and numerous, so tick labels start below the Makie default; anything
# passed in axis_kwargs still wins, since it is splatted last.
const _TICKLABEL_DEFAULTS = (; xticklabelsize=9, yticklabelsize=9)

_is_categorical(column) = !(eltype(column) <: Real)
_marginal(column) = _is_categorical(column) ? AlgebraOfGraphics.frequency() : AlgebraOfGraphics.histogram()

# Categorical levels are plotted by their `string`, so values that define no `isless`
# (functions, say) still sort into the positions AlgebraOfGraphics assigns them.
function _panel(completed, p)
    _is_categorical(completed[!, p]) || return completed, NamedTuple()
    labels = string.(completed[!, p])
    frame = copy(completed)
    frame[!, p] = labels
    levels = sort(unique(labels))
    return frame, (; xticks=(1:length(levels), levels))
end

function _draw_marginal_scatter!(fig, row, col, completed, p; axis_kwargs, scatter_kwargs, histogram_kwargs)
    frame, ticks = _panel(completed, p)
    gl = fig[row, col] = GridLayout()
    ax_top = Axis(gl[1, 1]; _TICKLABEL_DEFAULTS..., axis_kwargs...)
    ax_main = Axis(gl[2, 1]; xlabel=string(p), ylabel="value", ticks..., _TICKLABEL_DEFAULTS..., axis_kwargs...)
    ax_right = Axis(gl[2, 2]; _TICKLABEL_DEFAULTS..., axis_kwargs...)
    draw!(ax_top, data(frame) * mapping(p) * _marginal(frame[!, p]) * visual(BarPlot; histogram_kwargs...))
    draw!(ax_main, data(frame) * mapping(p, :value) * visual(Scatter; scatter_kwargs...))
    draw!(ax_right, data(completed) * mapping(:value) * AlgebraOfGraphics.histogram() * visual(BarPlot; direction=:x, histogram_kwargs...))
    linkxaxes!(ax_top, ax_main)
    linkyaxes!(ax_right, ax_main)
    hidedecorations!(ax_top, grid=false)
    hidedecorations!(ax_right, grid=false)
    colsize!(gl, 2, Relative(0.2))
    rowsize!(gl, 1, Relative(0.2))
    return nothing
end

function _draw_value_over_id!(fig, row, col, completed; axis_kwargs, scatter_kwargs)
    ax = Axis(fig[row, col]; xlabel="id", ylabel="value", title="obj. value over id", _TICKLABEL_DEFAULTS..., axis_kwargs...)
    draw!(ax, data(completed) * mapping(:id, :value) * visual(Scatter; scatter_kwargs...))
    return nothing
end

function _draw_running_min!(fig, row, col, completed; axis_kwargs, line_kwargs)
    ax = Axis(fig[row, col]; xlabel="id", ylabel="best value so far", title="min. over trials", _TICKLABEL_DEFAULTS..., axis_kwargs...)
    lines!(ax, completed.id, accumulate(min, completed.value); line_kwargs...)
    return nothing
end

function BigHOPs.summaryplot(ho::BigHOPs.Hyperoptimizer; figure=NamedTuple(), axis=NamedTuple(),
                            scatter=NamedTuple(), histogram=NamedTuple(), line=NamedTuple())
    isempty(ho.completed) &&
        error("summaryplot: this Hyperoptimizer has no completed trials")
    completed = sort(BigHOPs.DataFrames.DataFrame(ho)[ho.completed, :], :id)

    panels = Any[(fig, row, col) -> _draw_marginal_scatter!(fig, row, col, completed, p; axis_kwargs=axis, scatter_kwargs=scatter,
                                                            histogram_kwargs=histogram) for p in ho.params]
    push!(panels, (fig, row, col) -> _draw_value_over_id!(fig, row, col, completed; axis_kwargs=axis, scatter_kwargs=scatter))
    push!(panels, (fig, row, col) -> _draw_running_min!(fig, row, col, completed; axis_kwargs=axis, line_kwargs=line))

    # As square as possible (e.g. 4 -> 2x2, 6 -> 2x3), rather than one long column.
    total = length(panels)
    nrows = max(1, floor(Int, sqrt(total)))
    ncols = cld(total, nrows)

    fig = Figure(; size=(400 * ncols, 400 * nrows), figure...)
    for (i, panel) in enumerate(panels)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        panel(fig, row, col)
    end

    return fig
end

end # module
