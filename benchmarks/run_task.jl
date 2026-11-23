using JLD2

include("tasks.jl")

const BENCHMARK_FILES = Dict("counting_ones" => "benchmark_code/counting_ones/counting_ones.jl", "nn" => "benchmark_code/nn/nn.jl")
const BENCHMARK = ARGS[1]
const JOB = parse(Int, ARGS[2])

function job_tasks(benchmark, job)
    per_job = TASKS_PER_JOB[benchmark]
    return TASKS[benchmark][((job - 1) * per_job + 1):min(job * per_job, length(TASKS[benchmark]))]
end

include(BENCHMARK_FILES[BENCHMARK])

for task in job_tasks(BENCHMARK, JOB)
    isfile(task.path) && continue
    result = run_task(task)
    mkpath(dirname(task.path))
    JLD2.jldsave(task.path * ".tmp"; result...)
    mv(task.path * ".tmp", task.path; force=true)
end
