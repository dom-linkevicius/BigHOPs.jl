using Distributed

const WORKERS = parse(Int, get(ENV, "BIGHO_COUNTING_ONES_WORKERS", "0"))
const COUNTING_ONES = joinpath(@__DIR__, "counting_ones.jl")

addprocs(WORKERS; exeflags=["--project=$(Base.active_project())", "-t1"])
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
@everywhere include($COUNTING_ONES)

const REPEATS = parse(Int, get(ENV, "BIGHO_COUNTING_ONES_REPEATS", "50"))

const RUNS = (
    (ncat=32, ncont=32, samplers=(Random=1e6, Hyperband=1e4, DEHB=1e4, GP=3000.0)),
    (ncat=8, ncont=8, samplers=(Random=1e4, Hyperband=1e4, DEHB=1e4, GP=3000.0)),
)

println("Counting Ones: per-iteration budget=$ITERATION_BUDGET, repeats=$REPEATS, workers=$WORKERS, results=$RESULTS_DIR")
for run in RUNS, (name, target) in pairs(run.samplers)
    run_counting_ones(String(name), Float64(target); ncat=run.ncat, ncont=run.ncont, repeats=REPEATS)
end
