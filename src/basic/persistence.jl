"""
    load_hyperoptimizer(objective, path) -> Hyperoptimizer

Restore a checkpoint written by `save_hyperoptimizer` or `run!`, supplying `objective`, which checkpoints do not store. Trials still running when the checkpoint was written are marked `Abandoned`.
"""
function load_hyperoptimizer(objective, path::AbstractString)
    saved = JLD2.jldopen(file -> file["ho"], path, "r")
    runs = saved.runs
    abandoned = 0
    for (i, entry) in enumerate(runs)
        if entry.status === Pending
            runs[i] = _with_result(entry, Abandoned, missing, entry.post_artefact)
            abandoned += 1
        end
    end
    abandoned > 0 && @warn "load_hyperoptimizer: $abandoned trial(s) were still in flight when this checkpoint was written, so their outcome was never recorded and never can be; marking them Abandoned so the run can continue"
    return Hyperoptimizer(saved.params, saved.candidates, saved.sampler, objective, saved.n,
                           runs, saved.completed, saved.n_pending - abandoned, saved.status,
                           saved.best_min_id, ReentrantLock())
end

"""
    save_hyperoptimizer(ho, path)

Write `ho`, without its objective, to `path` as a JLD2 checkpoint. It is written to a temporary file first and then moved into place, so a crash mid-save leaves the previous checkpoint intact.
"""
function save_hyperoptimizer(ho::Hyperoptimizer, path::AbstractString)
    sanitized = Hyperoptimizer(ho.params, ho.candidates, ho.sampler, nothing, ho.n,
                                ho.runs, ho.completed, ho.n_pending, ho.status,
                                ho.best_min_id, ReentrantLock())
    tmp_path = path * ".tmp"
    JLD2.jldopen(tmp_path, "w") do file
        file["ho"] = sanitized
    end
    mv(tmp_path, path; force=true)
    return nothing
end
