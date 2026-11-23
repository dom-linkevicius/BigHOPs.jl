using JLD2

include("tasks.jl")

const BENCHMARK_FILES = Dict("counting_ones" => "benchmark_code/counting_ones/counting_ones.jl", "nn" => "benchmark_code/nn/nn.jl")
const BENCHMARK = ARGS[1]
const TASK = TASKS[BENCHMARK][parse(Int, ARGS[2])]

include(BENCHMARK_FILES[BENCHMARK])

result = run_task(TASK)
mkpath(dirname(TASK.path))
JLD2.jldsave(TASK.path * ".tmp"; result...)
mv(TASK.path * ".tmp", TASK.path; force=true)
