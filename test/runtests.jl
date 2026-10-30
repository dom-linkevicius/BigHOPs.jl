using Test, Random, Logging, Distributed
using StableRNGs: StableRNG
using BigHOPs
using DataFrames
using CairoMakie
using AlgebraOfGraphics
using AbstractGPs

const CONCERNS = Dict(
    "basic" => ["basic/domains.jl", "basic/artefacts.jl", "basic/failures.jl", "basic/manual.jl", "basic/persistence.jl", "basic/progress.jl", "basic/dataframe.jl", "basic/plotting.jl"],
    "samplers" => ["samplers/random.jl", "samplers/lhs.jl", "samplers/sh_based.jl", "samplers/dehb.jl", "samplers/gp.jl"],
    "serial" => ["executors/serial.jl"],
    "threaded" => ["executors/threaded.jl"],
    "distributed" => ["executors/distributed_queue.jl", "executors/equivalence.jl"],
)
const MULTITHREADED_CONCERNS = ("threaded", "distributed")

concerns_to_run = isempty(ARGS) ? sort(collect(keys(CONCERNS))) : ARGS

for concern in concerns_to_run
    haskey(CONCERNS, concern) ||
        error("unknown concern \"$concern\" -- choose from: $(join(sort(collect(keys(CONCERNS))), ", "))")
end

skipped = Threads.nthreads() == 1 ? filter(in(MULTITHREADED_CONCERNS), concerns_to_run) : String[]
isempty(skipped) || @warn "Julia was started with a single thread, so the $(join(skipped, " and ")) tests cannot show real parallelism; skipping them. Start Julia with more than one thread to run them, e.g. Pkg.test(\"BigHOPs\"; julia_args=[\"--threads=auto\"])."

for concern in concerns_to_run
    concern in skipped && continue
    for file in CONCERNS[concern]
        include(file)
    end
end
