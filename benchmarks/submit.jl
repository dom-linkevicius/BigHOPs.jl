using Pkg

ENV["JULIA_CPU_TARGET"] = "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)"
Pkg.resolve()
Pkg.instantiate(; allow_autoprecomp=false)

include("tasks.jl")

const RERUN = !isempty(ARGS)
const BENCHMARKS = isempty(ARGS) || ARGS == ["all"] ? sort(collect(keys(TASKS))) : ARGS

all(in(keys(TASKS)), BENCHMARKS) ||
    error("unknown benchmark in $BENCHMARKS; pass any of $(sort(collect(keys(TASKS)))), \"all\", or nothing")

function precompile_dependency()
    stale_cache, cachepath_cache = Dict{Base.StaleCacheKey,Bool}(), Dict{Base.PkgId,Vector{String}}()
    ids = [Base.PkgId(uuid, info.name) for (uuid, info) in Pkg.dependencies()]
    all(id -> Base.in_sysimage(id) || Base.isprecompiled(id; stale_cache, cachepath_cache), ids) && return String[]
    mkpath(joinpath(@__DIR__, "results", "logs"))
    job = first(split(readchomp(Cmd(`sbatch --parsable precompile.sbatch`; dir=@__DIR__)), ';'))
    println("precompiling in job $job, benchmark jobs wait for it")
    return ["--dependency=afterok:$job", "--kill-on-invalid-dep=yes"]
end

const DEPENDENCY = precompile_dependency()

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
        run(Cmd(`sbatch $DEPENDENCY --array=$(array_spec(jobs)) $sbatch`; dir=@__DIR__))
    end
    println("$benchmark: submitted $(length(pending)) of $(length(tasks)) tasks")
end
