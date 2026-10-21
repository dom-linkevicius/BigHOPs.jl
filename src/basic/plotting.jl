"""
    summaryplot(ho::Hyperoptimizer; figure_kwargs=(;), axis_kwargs=(;), scatter_kwargs=(;), histogram_kwargs=(;), line_kwargs=(;))

Plot the completed trials: objective value against each hyperparameter with marginal histograms, value against trial id, and the running minimum. Needs `CairoMakie` and `AlgebraOfGraphics` loaded; each keyword is passed through to the corresponding Makie call.
"""
function summaryplot end
