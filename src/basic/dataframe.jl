"""
    DataFrame(ho::Hyperoptimizer) -> DataFrame

One row per trial, including failed and pending ones, with columns `id`, `status`, each hyperparameter, `value`, `error`, `pre_artefact`, `post_artefact` and `metadata`, a copy of the trial's metadata `Dict`.
"""
function DataFrames.DataFrame(ho::Hyperoptimizer)
    rows = [merge((id=e.id, status=e.status), e.params,
                   (value=e.value, error=e.error, pre_artefact=e.pre_artefact, post_artefact=e.post_artefact, metadata=copy(e.metadata)))
            for e in ho.runs]
    return DataFrames.DataFrame(rows)
end
