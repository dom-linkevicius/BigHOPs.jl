using JLD2

include("tasks.jl")

const BENCHMARK_FILES = Dict("counting_ones" => "benchmark_code/counting_ones/counting_ones.jl", "nn" => "benchmark_code/nn/nn.jl")
const BENCHMARK = ARGS[1]
const JOB = parse(Int, ARGS[2])

include(BENCHMARK_FILES[BENCHMARK])

for task in filter(task -> task.job == JOB, TASKS[BENCHMARK])
    isfile(task.path) && continue
    result = run_task(task)
    mkpath(dirname(task.path))
    JLD2.jldsave(task.path * ".tmp"; result...)
    mv(task.path * ".tmp", task.path; force=true)
end
