using Pkg

Pkg.instantiate()
Pkg.precompile()

include("tasks.jl")

const RERUN = !isempty(ARGS)
const BENCHMARKS = isempty(ARGS) || ARGS == ["all"] ? sort(collect(keys(TASKS))) : ARGS

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

for benchmark in BENCHMARKS
    mkpath(joinpath(@__DIR__, "results", benchmark, "logs"))
    tasks = TASKS[benchmark]
    RERUN && foreach(task -> rm(task.path; force=true), tasks)
    pending = [i for (i, task) in enumerate(tasks) if !isfile(task.path)]
    for sbatch in unique(tasks[i].sbatch for i in pending)
        jobs = unique(tasks[i].job for i in pending if tasks[i].sbatch == sbatch)
        run(Cmd(`sbatch --array=$(array_spec(jobs)) $sbatch`; dir=@__DIR__))
    end
    println("$benchmark: submitted $(length(pending)) of $(length(tasks)) tasks")
end
