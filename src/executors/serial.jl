"""
    Serial()

Evaluate trials one at a time, in the calling task.
"""
mutable struct Serial <: AbstractExecutor
    buffer::Vector{Tuple{RunEntry,Any}}
end
Serial() = Serial(Tuple{RunEntry,Any}[])

start!(::Serial, ho) = nothing
shutdown!(::Serial) = nothing

function submit!(executor::Serial, entry::RunEntry, f)
    push!(executor.buffer, (entry, safe_call(f, entry.params, entry.pre_artefact)))
    return nothing
end

function poll(executor::Serial)
    out = copy(executor.buffer)
    empty!(executor.buffer)
    return out
end

capacity(executor::Serial) = isempty(executor.buffer) ? 1 : 0
