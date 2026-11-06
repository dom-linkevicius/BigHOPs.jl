using BigHOPs
using JLD2

include("nn_objective.jl")

_log(msg) = (println(msg); flush(stdout))

const RESULTS_DIR = joinpath(@__DIR__, "nn_results")

results_path(name, ex_name) = joinpath(RESULTS_DIR, "$(name)_$(ex_name).jld2")

const CANDIDATES = (
    lr=Continuous(-5, -1; transform=x -> 10.0^x),
    n_dense_layers=Ordinal([1, 2, 3]),
    hidden=Ordinal([8, 16, 32, 64]),
    activation=Nominal([tanh, relu]),
    reg=Continuous(-6, -1; transform=x -> 10.0^x),
)

const ETA = 3

const N_TRIALS = DEPLOY ? BigHOPs._total_draws(R_MAX, R_MIN, ETA) : 4
const REGRET_REPEATS = parse(Int, get(ENV, "BIGHO_BENCHMARK_REPEATS", DEPLOY ? "20" : "2"))

if DEPLOY
    smax = BigHOPs._smax(R_MAX, R_MIN, ETA)
    total_resource_units = sum(
        BigHOPs._capacity(R_MAX, R_MIN, ETA, k, i) * (BigHOPs._resource(R_MAX, R_MIN, ETA, k, i) - (i > 1 ? BigHOPs._resource(R_MAX, R_MIN, ETA, k, i - 1) : 0))
        for k in 1:(smax+1) for i in 1:BigHOPs._n_rungs(R_MAX, R_MIN, ETA, k)
    )
    hyperband_total_epochs = total_resource_units * EPOCHS_PER_RESOURCE
    RANDOM_FULL_EPOCHS[] = round(Int, hyperband_total_epochs / N_TRIALS)
    random_total_epochs = RANDOM_FULL_EPOCHS[] * N_TRIALS
    _log("Matched Random/LHS per-trial epochs to Hyperband/ASHA's total budget: random_full_epochs=$(RANDOM_FULL_EPOCHS[]) random_total_epochs=$random_total_epochs hyperband_total_epochs=$hyperband_total_epochs")
end

const PLAIN_OBJ = Stateful(nn_objective)
const SH_OBJ = Stateful(nn_objective_stateful)

const SAMPLER_NAMES = ("Random", "LHS", "Hyperband", "ASHA", "DEHB", "Sobol")
const EXECUTORS = (Serial=Serial(), Threaded=Threaded())

const SHA_INNER = RandomSampler()

const MAKE_HYPEROPTIMIZER = Dict(
    "Random" => () -> Hyperoptimizer(PLAIN_OBJ, CANDIDATES; sampler=RandomSampler(), n=N_TRIALS),
    "LHS" => () -> Hyperoptimizer(PLAIN_OBJ, CANDIDATES; sampler=LHSampler(), n=N_TRIALS),
    "Sobol" => () -> Hyperoptimizer(PLAIN_OBJ, CANDIDATES; sampler=SobolSampler(), n=N_TRIALS),
    "Hyperband" => () -> Hyperoptimizer(SH_OBJ, CANDIDATES, Hyperband(R=R_MAX, η=ETA, r_min=R_MIN, inner=SHA_INNER)),
    "ASHA" => () -> Hyperoptimizer(SH_OBJ, CANDIDATES, ASHA(R=R_MAX, η=ETA, r_min=R_MIN, inner=SHA_INNER)),
    "DEHB" => () -> Hyperoptimizer(SH_OBJ, CANDIDATES, DEHB(R=R_MAX, η=ETA, r_min=R_MIN)),
)

_log("Warming up (JIT compilation)...")
const WARMUP_OBJ = Stateful(_warmup_objective)
const WARMUP_SH_OBJ = Stateful(_warmup_objective_stateful)
for executor in EXECUTORS, sampler in (RandomSampler(), LHSampler(), SobolSampler())
    run!(Hyperoptimizer(WARMUP_OBJ, CANDIDATES; sampler=sampler, n=2); executor=executor, show_progress=false)
end
for sh_sampler in (Hyperband(R=1, η=3, r_min=1), ASHA(R=1, η=3, r_min=1), DEHB(R=3, η=3, r_min=1))
    run!(Hyperoptimizer(WARMUP_SH_OBJ, CANDIDATES, sh_sampler); executor=Serial(), show_progress=false)
end

_finished_at(entry) = entry.post_artefact[4]

"""
    _timed_results(ho, t0)
"""
function _timed_results(ho::Hyperoptimizer, t0::Float64)
    completed = filter(e -> !ismissing(e.value), ho.runs)
    pairs = [(_finished_at(e) - t0, e.value) for e in completed]
    return sort(pairs; by=first)
end

function _running_min_curve(pairs)
    times, values = first.(pairs), last.(pairs)
    best = accumulate(min, values)
    return times, best
end

function collect_results()
    mkpath(RESULTS_DIR)
    metadata = (n_trials=N_TRIALS, r_max=R_MAX, deploy=DEPLOY, nthreads=Threads.nthreads(),
                sha_inner_sampler=nameof(typeof(SHA_INNER)))
    total_combos = REGRET_REPEATS * length(EXECUTORS) * length(SAMPLER_NAMES)
    combo = 0

    for repeat in 1:REGRET_REPEATS, ex_name in keys(EXECUTORS), name in SAMPLER_NAMES
        combo += 1
        path = results_path(name, ex_name)
        if !isfile(path)
            JLD2.jldsave(path; metadata)
        end
        JLD2.jldopen(file -> haskey(file, "repeat_$repeat"), path, "r") && continue
        _log("[$combo/$total_combos] repeat=$repeat sampler=$name executor=$ex_name -- starting")
        ho = MAKE_HYPEROPTIMIZER[name]()
        t0 = time()
        run!(ho; executor=EXECUTORS[ex_name], show_progress=false)
        pairs = _timed_results(ho, t0)
        elapsed = time() - t0
        best = isempty(pairs) ? NaN : minimum(last, pairs)
        _log("[$combo/$total_combos] repeat=$repeat sampler=$name executor=$ex_name -- done elapsed=$elapsed best=$best")
        JLD2.jldopen(path, "a+") do file
            file["repeat_$repeat"] = _running_min_curve(pairs)
        end
    end
end

_log("Collecting wall-clock regret comparison results...")
collect_results()
_log("Results written to $RESULTS_DIR")
