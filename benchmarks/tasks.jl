const COUNTING_ONES_DIR = joinpath(@__DIR__, "results", "counting_ones")
const COUNTING_ONES_REPEATS = parse(Int, get(ENV, "BIGHOPS_COUNTING_ONES_REPEATS", "512"))
const COUNTING_ONES_GP_REPEATS = min(COUNTING_ONES_REPEATS, 50)
const COUNTING_ONES_RUNS = (
    (ncat=32, ncont=32, samplers=(Random=1e6, Hyperband=1e4, DEHB=1e4, BOHB=1e4, GP=1e4)),
    (ncat=8, ncont=8, samplers=(Random=1e4, Hyperband=1e4, DEHB=1e4, BOHB=1e4, GP=1e4)),
)
const COUNTING_ONES_TASKS = [(ncat=run.ncat, ncont=run.ncont, sampler=String(name), target=Float64(target), repeat=repeat,
                              sbatch="benchmark_code/counting_ones/counting_ones$(name == :GP ? "_gp" : "").sbatch",
                              path=joinpath(COUNTING_ONES_DIR, "counting_ones_$(run.ncat)cat_$(run.ncont)cont_$(name)", "repeat_$repeat.jld2"))
                             for run in COUNTING_ONES_RUNS for (name, target) in pairs(run.samplers)
                             for repeat in 1:(name == :GP ? COUNTING_ONES_GP_REPEATS : COUNTING_ONES_REPEATS)]

const NN_DIR = joinpath(@__DIR__, "results", "nn")
const NN_REPEATS = parse(Int, get(ENV, "BIGHOPS_BENCHMARK_REPEATS", "50"))
const NN_SAMPLERS = ("Random", "Hyperband", "ASHA", "DEHB", "BOHB", "GP")
const NN_EXECUTORS = ("Threaded", "Serial")
const NN_TASKS = [(sampler=name, executor=ex_name, repeat=repeat, sbatch="benchmark_code/nn/nn_$(lowercase(ex_name)).sbatch",
                   path=joinpath(NN_DIR, "$(name)_$(ex_name)", "repeat_$repeat.jld2"))
                  for ex_name in NN_EXECUTORS for repeat in 1:NN_REPEATS for name in NN_SAMPLERS]

const TASKS_PER_JOB = Dict("benchmark_code/counting_ones/counting_ones.sbatch" => 4)

function with_jobs(tasks)
    jobbed = NamedTuple[]
    for sbatch in unique(task.sbatch for task in tasks)
        per_job, previous_jobs = get(TASKS_PER_JOB, sbatch, 1), isempty(jobbed) ? 0 : jobbed[end].job
        group = filter(task -> task.sbatch == sbatch, tasks)
        append!(jobbed, [(; task..., job=previous_jobs + cld(i, per_job)) for (i, task) in enumerate(group)])
    end
    return jobbed
end

const TASKS = Dict("counting_ones" => with_jobs(COUNTING_ONES_TASKS), "nn" => with_jobs(NN_TASKS))
