"""
    DataFrame(ho::Hyperoptimizer) -> DataFrame

One row per trial, including failed and pending ones, with columns `id`, `status`, each hyperparameter, `value`, `error`, `pre_artefact` and `post_artefact`.
"""
function DataFrames.DataFrame(ho::Hyperoptimizer)
    rows = [merge((id=e.id, status=e.status), e.params,
                   (value=e.value, error=e.error, pre_artefact=e.pre_artefact, post_artefact=e.post_artefact))
            for e in ho.runs]
    return DataFrames.DataFrame(rows)
end
