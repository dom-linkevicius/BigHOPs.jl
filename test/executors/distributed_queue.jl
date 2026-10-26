mutable struct SequentialSampler <: BigHOPs.Sampler
    n_calls::Int
end
SequentialSampler() = SequentialSampler(0)
function (s::SequentialSampler)(candidates, runs)
    s.n_calls += 1
    return [(s.n_calls - 0.5) / length(d) for d in candidates]
end
BigHOPs.init(s::SequentialSampler, candidates, n) = s
BigHOPs.exhausted(::SequentialSampler, ho) = false
BigHOPs.blocked(::SequentialSampler, ho) = false
BigHOPs.on_tell!(::SequentialSampler, runs, entry) = nothing
BigHOPs.create_run_entry(::SequentialSampler, ho, id, params, unit) = BigHOPs.RunEntry(id, params, unit)

@testset "DistributedQueue executor" begin
    @info "Testing DistributedQueue executor"

    @test_throws ArgumentError DistributedQueue(0; spawn_worker=() -> error("should not be called"))
    @test_throws ArgumentError DistributedQueue(-1; spawn_worker=() -> error("should not be called"))

    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=0)
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=-1)
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=NaN)

    @test_throws UndefKeywordError DistributedQueue()
    @test_throws UndefKeywordError DistributedQueue(4)

    test_spawn_worker() = first(addprocs(1))
    function test_setup_worker(pid)
        Distributed.remotecall_eval(Main, [pid], :(begin
            using BigHOPs
            dq_square(p) = p.a^2
            dq_slow(p) = (sleep(0.5); p.a^2)
        end))
        return nothing
    end
    @everywhere using BigHOPs
    @everywhere dq_square(p) = p.a^2
    @everywhere dq_slow(p) = (sleep(0.5); p.a^2)

    max_concurrency = 3
    try
        ex = DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker)
        BigHOPs.start!(ex, nothing)
        @test BigHOPs.capacity(ex) == max_concurrency
        entry1 = BigHOPs.RunEntry(1, (a=1,), Float64[])
        BigHOPs.submit!(ex, entry1, dq_square)
        @test BigHOPs.capacity(ex) == max_concurrency - 1
        entry2 = BigHOPs.RunEntry(2, (a=2,), Float64[])
        BigHOPs.submit!(ex, entry2, dq_square)
        @test BigHOPs.capacity(ex) == max_concurrency - 2
        out = BigHOPs.poll(ex)
        append!(out, BigHOPs.poll(ex))
        @test length(out) == 2
        @test BigHOPs.capacity(ex) == max_concurrency
        @test Set(e.id for (e, _) in out) == Set([1, 2])
        BigHOPs.shutdown!(ex)

        ex_batch = DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker)
        BigHOPs.start!(ex_batch, nothing)
        entryA = BigHOPs.RunEntry(1, (a=1,), Float64[])
        entryB = BigHOPs.RunEntry(2, (a=2,), Float64[])
        entryC = BigHOPs.RunEntry(3, (a=3,), Float64[])
        put!(ex_batch.results, (entryA, BigHOPs.ObjectiveOutcome(10, nothing)))
        put!(ex_batch.results, (entryB, InterruptException()))
        put!(ex_batch.results, (entryC, BigHOPs.ObjectiveOutcome(30, nothing)))
        ex_batch.in_flight = 3
        out_batch = BigHOPs.poll(ex_batch)
        @test Set(e.id for (e, _) in out_batch) == Set([1, 3])
        @test BigHOPs.capacity(ex_batch) == 0
        @test_throws InterruptException BigHOPs.poll(ex_batch)
        BigHOPs.shutdown!(ex_batch)

        shutdown_spawned = Int[]
        function shutdown_spawn_worker()
            pid = first(addprocs(1))
            push!(shutdown_spawned, pid)
            return pid
        end
        shutdown_setup_worker(pid) = Distributed.remotecall_eval(Main, [pid], :(using BigHOPs))
        ex_shutdown = DistributedQueue(2; spawn_worker=shutdown_spawn_worker, setup_worker=shutdown_setup_worker)
        BigHOPs.start!(ex_shutdown, nothing)
        BigHOPs.submit!(ex_shutdown, BigHOPs.RunEntry(1, (a=1,), Float64[]), p -> p.a^2)
        BigHOPs.submit!(ex_shutdown, BigHOPs.RunEntry(2, (a=2,), Float64[]), p -> (sleep(30.0); p.a^2))
        sleep(5.0)
        elapsed = @elapsed BigHOPs.shutdown!(ex_shutdown)
        @test elapsed < 10.0
        out_shutdown = BigHOPs.poll(ex_shutdown)
        out_shutdown_by_id = Dict(e.id => outcome for (e, outcome) in out_shutdown)
        @test out_shutdown_by_id[1].value == 1
        @test out_shutdown_by_id[2] isa Exception
        @test length(shutdown_spawned) == 2
        @test isempty(intersect(shutdown_spawned, workers()))

        ex_task_failure = DistributedQueue(1; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker)
        BigHOPs.start!(ex_task_failure, nothing)
        failed_task = @async throw(InterruptException())
        try
            wait(failed_task)
        catch
        end
        @test istaskfailed(failed_task)
        push!(ex_task_failure.tasks, (0, failed_task))
        @test_throws TaskFailedException BigHOPs.shutdown!(ex_task_failure)

        setup_done = Ref(false)
        function interrupt_setup_worker(pid)
            test_setup_worker(pid)
            setup_done[] = true
        end
        ex_interrupt = DistributedQueue(1; spawn_worker=test_spawn_worker, setup_worker=interrupt_setup_worker)
        BigHOPs.start!(ex_interrupt, nothing)
        BigHOPs.submit!(ex_interrupt, BigHOPs.RunEntry(1, (a=1,), Float64[]), dq_slow)
        let waited = 0.0
            while !setup_done[] && waited < 60.0
                sleep(0.1)
                waited += 0.1
            end
        end
        @test setup_done[]
        sleep(0.1)
        schedule(ex_interrupt.tasks[1][2], InterruptException(); error=true)
        @test isempty(BigHOPs.poll(ex_interrupt))
        @test_throws InterruptException BigHOPs.poll(ex_interrupt)
        BigHOPs.shutdown!(ex_interrupt)

        ex_objective_interrupt = DistributedQueue(1; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker)
        BigHOPs.start!(ex_objective_interrupt, nothing)
        BigHOPs.submit!(ex_objective_interrupt, BigHOPs.RunEntry(1, (a=1,), Float64[]), p -> throw(InterruptException()))
        out_objective_interrupt = BigHOPs.poll(ex_objective_interrupt)
        @test length(out_objective_interrupt) == 1
        @test out_objective_interrupt[1][2] isa Exception
        BigHOPs.shutdown!(ex_objective_interrupt)

        ho = Hyperoptimizer(dq_square, (a=Ordinal(1:5),); n=20)
        run!(ho; executor=DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker))
        @test length(results(ho)) == 20
        @test all(e -> e.status == BigHOPs.Completed, ho.runs)
        @test minimum(ho) == 1

        n_death_trials = 4
        dying_id = 2
        death_spawned = Int[]
        function death_spawn_worker()
            pid = first(addprocs(1))
            push!(death_spawned, pid)
            return pid
        end
        death_setup_worker(pid) = Distributed.remotecall_eval(Main, [pid], :(using BigHOPs))
        dq_death_or_square = p -> p.a == dying_id ? exit() : p.a^2
        ex_death = DistributedQueue(n_death_trials; spawn_worker=death_spawn_worker, setup_worker=death_setup_worker)
        BigHOPs.start!(ex_death, nothing)
        for i in 1:n_death_trials
            BigHOPs.submit!(ex_death, BigHOPs.RunEntry(i, (a=i,), Float64[]), dq_death_or_square)
        end
        collected = Tuple{BigHOPs.RunEntry,Any}[]
        collect_task = @async begin
            while length(collected) < n_death_trials
                append!(collected, BigHOPs.poll(ex_death))
            end
        end
        let waited = 0.0
            while !istaskdone(collect_task) && waited < 60.0
                sleep(0.5)
                waited += 0.5
            end
        end
        @test istaskdone(collect_task)
        @test length(collected) == n_death_trials
        @test length(death_spawned) == n_death_trials
        @test length(unique(death_spawned)) == n_death_trials
        outcome_by_id = Dict(e.id => outcome for (e, outcome) in collected)
        @test outcome_by_id[dying_id] isa Exception
        for i in 1:n_death_trials
            i == dying_id && continue
            @test outcome_by_id[i].value == i^2
        end
        BigHOPs.shutdown!(ex_death)

        interrupt_others_spawned = Int[]
        function interrupt_others_spawn_worker()
            pid = first(addprocs(1))
            push!(interrupt_others_spawned, pid)
            return pid
        end
        interrupt_others_setup_worker(pid) = Distributed.remotecall_eval(Main, [pid], :(using BigHOPs))
        dq_interrupt_others = p -> (sleep(30.0); p.a^2)
        ex_interrupt_others = DistributedQueue(3; spawn_worker=interrupt_others_spawn_worker, setup_worker=interrupt_others_setup_worker)
        ho_dq_interrupt = Hyperoptimizer(dq_interrupt_others, (a=Nominal([1, 2, 3]),);
                                          sampler=SequentialSampler(), n=3)
        run_task = @async run!(ho_dq_interrupt; executor=ex_interrupt_others)
        let waited = 0.0
            while length(interrupt_others_spawned) < 3 && waited < 60.0
                sleep(0.1)
                waited += 0.1
            end
        end
        @test length(interrupt_others_spawned) == 3
        schedule(ex_interrupt_others.tasks[1][2], InterruptException(); error=true)
        caught = nothing
        try
            wait(run_task)
        catch e
            caught = e
        end
        @test caught isa TaskFailedException
        @test caught.task.result isa InterruptException
        @test ho_dq_interrupt.status == BigHOPs.Errored
        @test ho_dq_interrupt.n_pending == 0
        @test length(ho_dq_interrupt.runs) == 3
        @test count(e -> e.status == BigHOPs.Abandoned, ho_dq_interrupt.runs) == 3
        @test length(interrupt_others_spawned) == 3
        @test isempty(intersect(interrupt_others_spawned, workers()))

        spawn_fail_spawned = Int[]
        function spawn_fail_spawn_worker()
            pid = first(addprocs(1))
            push!(spawn_fail_spawned, pid)
            return pid
        end
        spawn_fail_setup_worker(pid) = error("simulated cluster/remotecall_eval setup failure")
        ex_spawn_fail = DistributedQueue(1; spawn_worker=spawn_fail_spawn_worker, setup_worker=spawn_fail_setup_worker)
        BigHOPs.start!(ex_spawn_fail, nothing)
        BigHOPs.submit!(ex_spawn_fail, BigHOPs.RunEntry(1, (a=1,), Float64[]), dq_square)
        out_spawn_fail = BigHOPs.poll(ex_spawn_fail)
        @test length(out_spawn_fail) == 1
        @test out_spawn_fail[1][2] isa Exception
        BigHOPs.shutdown!(ex_spawn_fail)
        @test length(spawn_fail_spawned) == 1
        @test isempty(intersect(spawn_fail_spawned, workers()))
    finally
        rmprocs(filter(!=(1), workers()))
    end
end

@testset "DistributedQueue max_single_runtime" begin
    @info "Testing that DistributedQueue kills a trial exceeding max_single_runtime and marks it TimedOut"

    no_spawn() = error("should not be called")
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=no_spawn, max_single_runtime=0)
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=no_spawn, max_single_runtime=-1)
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=no_spawn, max_single_runtime=NaN)
    @test DistributedQueue(1; spawn_worker=no_spawn).max_single_runtime == Inf

    entry = BigHOPs.RunEntry(1, (a=1,), Float64[])
    scaled = DistributedQueue(1; spawn_worker=no_spawn, max_single_runtime=p -> 30 + 20 * p.r)
    @test BigHOPs._trial_limit(scaled, BigHOPs.RunEntry(1, (r=3, a=1), Float64[])) == 90
    @test BigHOPs._trial_limit(DistributedQueue(1; spawn_worker=no_spawn, max_single_runtime=5.0), entry) == 5.0
    for bad in (_ -> 0, _ -> -1.0, _ -> NaN, _ -> "soon")
        @test_throws ArgumentError BigHOPs._trial_limit(DistributedQueue(1; spawn_worker=no_spawn, max_single_runtime=bad), entry)
    end
    @test_throws ErrorException BigHOPs._trial_limit(DistributedQueue(1; spawn_worker=no_spawn, max_single_runtime=_ -> error("no limit")), entry)

    bad_limit = DistributedQueue(1; spawn_worker=no_spawn, max_single_runtime=_ -> -1.0)
    BigHOPs.start!(bad_limit, nothing)
    @test_logs (:warn, r"could not give this trial a limit") BigHOPs.submit!(bad_limit, entry, _ -> 0.0)
    rejected = BigHOPs.poll(bad_limit)
    @test length(rejected) == 1
    @test rejected[1][2] isa ArgumentError
    @test BigHOPs.capacity(bad_limit) == 1

    unresolved = Distributed.Future()
    @async begin
        sleep(2.0)
        put!(unresolved, 42.0)
    end
    late = @test_logs (:warn, r"could not stop timed-out worker") BigHOPs._fetch_within(0.5, unresolved, typemax(Int), 30)
    @test late == 42.0

    try
        idle = first(addprocs(1))
        BigHOPs._kill_worker(idle, nothing, 30)
        @test idle ∉ workers()

        dying = first(addprocs(1))
        @test_throws ProcessExitedException BigHOPs._fetch_within(60.0, remotecall(exit, dying), dying, 30)
    finally
        rmprocs(filter(!=(1), workers()))
    end

    @everywhere function dq_spin(p)
        p.a == 1 && return 1.0
        s = 0.0
        while true
            s += sin(s) + 1.0
        end
        return s
    end
    spin_spawned = Int[]
    function spin_spawn_worker()
        pid = first(addprocs(1))
        push!(spin_spawned, pid)
        return pid
    end
    spin_setup_worker(pid) = Distributed.remotecall_eval(Main, [pid], :(begin
        using BigHOPs
        function dq_spin(p)
            p.a == 1 && return 1.0
            s = 0.0
            while true
                s += sin(s) + 1.0
            end
            return s
        end
    end))

    limit = 15.0
    try
        ex = DistributedQueue(2; spawn_worker=spin_spawn_worker, setup_worker=spin_setup_worker,
                              max_single_runtime=p -> p.a == 1 ? 60.0 : limit)
        ho = Hyperoptimizer(dq_spin, (a=Nominal([1, 2]),); sampler=SequentialSampler(), n=2)
        @test_logs (:warn, r"max_single_runtime") match_mode = :any run!(ho; executor=ex, show_progress=false)
        @test ho.status == BigHOPs.Finished
        @test ho.n_pending == 0
        @test ho.runs[1].status == BigHOPs.Completed
        @test ho.runs[1].value == 1.0
        @test ho.runs[2].status == BigHOPs.TimedOut
        @test ho.runs[2].error == BigHOPs.TrialTimeout(limit)
        @test length(spin_spawned) == 2
        @test isempty(intersect(spin_spawned, workers()))
    finally
        rmprocs(filter(!=(1), workers()))
    end
end
