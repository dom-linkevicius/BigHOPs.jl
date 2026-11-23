"""
    AbstractExecutor

Decides how `run!` evaluates trials. An executor implements `start!`, `submit!`, `poll`, `capacity` and `shutdown!`.
"""
abstract type AbstractExecutor end

"""
    start!(executor, ho)

Reset the executor at the start of a `run!`.
"""
function start! end

"""
    shutdown!(executor; interrupt=false)

Wait for or clean up in-flight work at the end of a `run!`, including when it fails. With
`interrupt=true`, `run!` has abandoned the trials still running, so don't wait for them.
"""
function shutdown! end

"""
    submit!(executor, entry::RunEntry, f)

Hand `entry` to the executor to evaluate with objective `f`; its outcome comes back from a later `poll`.
"""
function submit! end

"""
    poll(executor, timeout=Inf) -> Vector{Tuple{RunEntry,Any}}

Return the `(entry, outcome)` pairs finished since the last call, waiting up to `timeout`
seconds for at least one if any are still running.
"""
function poll end

"""
    capacity(executor) -> Int

How many more trials the executor can accept right now.
"""
function capacity end

"""
    safe_call(f, params, pre_artefact) -> value_or_exception

Call the objective and return any exception it throws instead of raising it, so one failing trial does not stop the search. `InterruptException` is still raised.
"""
function safe_call(f, params, pre_artefact)
    try
        call_objective(f, params, pre_artefact)
    catch e
        e isa InterruptException && rethrow()
        e
    end
end
