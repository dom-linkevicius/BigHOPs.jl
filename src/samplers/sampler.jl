"""
    Sampler

Proposes configurations to try. A sampler is called as `sampler(candidates, runs)`, returning one value in `[0, 1]` per candidate, and implements `init`, `create_run_entry`, `on_tell!`, `exhausted` and `blocked`.
"""
abstract type Sampler end

"""
    on_tell!(sampler, runs, entry)

Update the sampler after `entry` has been told.
"""
function on_tell! end

"""
    init(sampler, candidates, n) -> sampler

Prepare the sampler for a search of `n` trials over `candidates`, returning the sampler to use, which may be a new one.
"""
function init end

"""
    exhausted(sampler, ho) -> Bool

Whether the sampler can never propose another trial.
"""
function exhausted end

"""
    blocked(sampler, ho) -> Bool

Whether the sampler can propose nothing until pending trials are told, as when a synchronous rung is waiting to finish.
"""
function blocked end

"""
    create_run_entry(sampler, ho, id, params, unit_params) -> RunEntry

Build the `RunEntry` for a newly proposed trial, including any metadata the sampler needs.
"""
function create_run_entry end
