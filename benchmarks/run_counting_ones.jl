include("counting_ones.jl")

const REPEATS = parse(Int, get(ENV, "BIGHO_COUNTING_ONES_REPEATS", "50"))

const RUNS = (
    (ncat=32, ncont=32, samplers=(Random=1e6, Hyperband=1e4, DEHB=1e4)),
    (ncat=8, ncont=8, samplers=(Random=1e4, Hyperband=1e4, DEHB=1e4)),
)

println("Counting Ones: per-iteration budget=$ITERATION_BUDGET, repeats=$REPEATS, results=$RESULTS_DIR")
for run in RUNS, (name, target) in pairs(run.samplers)
    run_counting_ones(String(name), Float64(target); ncat=run.ncat, ncont=run.ncont, repeats=REPEATS)
end
