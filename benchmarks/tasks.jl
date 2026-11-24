const COUNTING_ONES_DIR = joinpath(@__DIR__, "results", "counting_ones")
const COUNTING_ONES_REPEATS = parse(Int, get(ENV, "BIGHOPS_COUNTING_ONES_REPEATS", "512"))
const COUNTING_ONES_RUNS = (
    (ncat=32, ncont=32, samplers=(Random=1e6, Hyperband=1e4, DEHB=1e4, BOHB=1e4)),
    (ncat=8, ncont=8, samplers=(Random=1e4, Hyperband=1e4, DEHB=1e4, BOHB=1e4)),
)
const COUNTING_ONES_TASKS = [(ncat=run.ncat, ncont=run.ncont, sampler=String(name), target=Float64(target), repeat=repeat, sbatch="benchmark_code/counting_ones/counting_ones.sbatch",
                              path=joinpath(COUNTING_ONES_DIR, "counting_ones_$(run.ncat)cat_$(run.ncont)cont_$(name)", "repeat_$repeat.jld2"))
                             for run in COUNTING_ONES_RUNS for (name, target) in pairs(run.samplers) for repeat in 1:COUNTING_ONES_REPEATS]

const NN_DIR = joinpath(@__DIR__, "results", "nn")
const NN_REPEATS = parse(Int, get(ENV, "BIGHOPS_BENCHMARK_REPEATS", "50"))
const NN_SAMPLERS = ("Random", "LHS", "Hyperband", "ASHA", "DEHB", "Sobol", "BOHB")
const NN_EXECUTORS = ("Threaded", "Serial")
const NN_TASKS = [(sampler=name, executor=ex_name, repeat=repeat, sbatch="benchmark_code/nn/nn_$(lowercase(ex_name)).sbatch",
                   path=joinpath(NN_DIR, "$(name)_$(ex_name)", "repeat_$repeat.jld2"))
                  for ex_name in NN_EXECUTORS for repeat in 1:NN_REPEATS for name in NN_SAMPLERS]

const TASKS = Dict("counting_ones" => COUNTING_ONES_TASKS, "nn" => NN_TASKS)
const TASKS_PER_JOB = Dict("counting_ones" => 4, "nn" => 1)
