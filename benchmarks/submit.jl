using Pkg

Pkg.instantiate()
Pkg.precompile()

include("tasks.jl")

const RERUN = !isempty(ARGS)
const BENCHMARKS = isempty(ARGS) || ARGS == ["all"] ? sort(collect(keys(TASKS))) : ARGS
const LOG_DIR = joinpath(@__DIR__, "slurm_logs")

all(in(keys(TASKS)), BENCHMARKS) ||
    error("unknown benchmark in $BENCHMARKS; pass any of $(sort(collect(keys(TASKS)))), \"all\", or nothing")

function array_spec(ids)
    runs = UnitRange{Int}[]
    for id in ids
        if !isempty(runs) && last(runs[end]) == id - 1
            runs[end] = first(runs[end]):id
        else
            push!(runs, id:id)
        end
    end
    return join((length(r) == 1 ? "$(first(r))" : "$(first(r))-$(last(r))" for r in runs), ",")
end

mkpath(LOG_DIR)
for benchmark in BENCHMARKS
    tasks = TASKS[benchmark]
    RERUN && foreach(task -> rm(task.path; force=true), tasks)
    pending = [i for (i, task) in enumerate(tasks) if !isfile(task.path)]
    for cpus in unique(tasks[i].cpus for i in pending)
        ids = filter(i -> tasks[i].cpus == cpus, pending)
        wrap = "julia --project=. --threads=$cpus run_task.jl $benchmark \$SLURM_ARRAY_TASK_ID"
        run(`sbatch --job-name=bigho_$benchmark --array=$(array_spec(ids)) --cpus-per-task=$cpus --chdir=$(@__DIR__)
                    --output=$(joinpath(LOG_DIR, "%x_%A_%a.log")) --wrap=$wrap`)
    end
    println("$benchmark: submitted $(length(pending)) of $(length(tasks)) tasks")
end
