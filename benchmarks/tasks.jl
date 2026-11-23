const DEPLOY = get(ENV, "BIGHO_BENCHMARK_DEPLOY", "false") == "true"

const COUNTING_ONES_DIR = joinpath(@__DIR__, "results", "counting_ones")
const COUNTING_ONES_REPEATS = parse(Int, get(ENV, "BIGHO_COUNTING_ONES_REPEATS", "50"))
const COUNTING_ONES_RUNS = (
    (ncat=32, ncont=32, samplers=(Random=1e6, Hyperband=1e4, DEHB=1e4, BOHB=1e4)),
    (ncat=8, ncont=8, samplers=(Random=1e4, Hyperband=1e4, DEHB=1e4, BOHB=1e4)),
)
const COUNTING_ONES_TASKS = [(ncat=run.ncat, ncont=run.ncont, sampler=String(name), target=Float64(target), repeat=repeat, cpus=1,
                              path=joinpath(COUNTING_ONES_DIR, "counting_ones_$(run.ncat)cat_$(run.ncont)cont_$(name)", "repeat_$repeat.jld2"))
                             for run in COUNTING_ONES_RUNS for (name, target) in pairs(run.samplers) for repeat in 1:COUNTING_ONES_REPEATS]

const NN_DIR = joinpath(@__DIR__, "results", "nn")
const NN_REPEATS = parse(Int, get(ENV, "BIGHO_BENCHMARK_REPEATS", DEPLOY ? "20" : "2"))
const NN_THREADS = parse(Int, get(ENV, "BIGHO_NN_THREADS", "64"))
const NN_SAMPLERS = ("Random", "LHS", "Hyperband", "ASHA", "DEHB", "Sobol", "BOHB")
const NN_EXECUTORS = ("Threaded", "Serial")
const NN_TASKS = [(sampler=name, executor=ex_name, repeat=repeat, cpus=ex_name == "Threaded" ? NN_THREADS : 1,
                   path=joinpath(NN_DIR, "$(name)_$(ex_name)", "repeat_$repeat.jld2"))
                  for ex_name in NN_EXECUTORS for repeat in 1:NN_REPEATS for name in NN_SAMPLERS]

const TASKS = Dict("counting_ones" => COUNTING_ONES_TASKS, "nn" => NN_TASKS)
