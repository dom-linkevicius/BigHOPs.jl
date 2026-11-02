include("counting_ones.jl")

const REPEATS = parse(Int, get(ENV, "BIGHO_COUNTING_ONES_REPEATS", "5"))
const CACHEDIR = get(() -> mktempdir(; cleanup=false), ENV, "BIGHO_COUNTING_ONES_CACHE")

const RUNS = (
    (ncat=32, ncont=32, samplers=(Random=1e6, Hyperband=1e4, DEHB=1e4)),
    (ncat=8, ncont=8, samplers=(Random=1e6, Hyperband=1e4, DEHB=1e4)),
)

println("Counting Ones: per-iteration budget=$ITERATION_BUDGET, repeats=$REPEATS, cache=$CACHEDIR")
for run in RUNS
    plot_counting_ones(; run..., repeats=REPEATS, cachedir=CACHEDIR)
    println("$(run.ncat) categorical + $(run.ncont) continuous: plot written to $OUTDIR")
end
