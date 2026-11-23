mutable struct BuggySampler <: BigHOPs.Sampler
    n_calls::Int
end
BuggySampler() = BuggySampler(0)
function (s::BuggySampler)(candidates, runs)
    s.n_calls += 1
    s.n_calls == 3 && error("sampler bug")
    return [(s.n_calls - 0.5) / length(d) for d in candidates]
end
BigHOPs.init(s::BuggySampler, candidates, n) = s
BigHOPs.exhausted(::BuggySampler, ho) = false
BigHOPs.blocked(::BuggySampler, ho) = false
BigHOPs.on_tell!(::BuggySampler, runs, entry) = nothing
BigHOPs.create_run_entry(::BuggySampler, ho, id, params, unit) = BigHOPs.RunEntry(id, params, unit)

@testset "Threaded executor" begin
    @info "Testing Threaded executor"

    @test_throws ArgumentError Threaded(0)
    @test_throws ArgumentError Threaded(-1)

    ex = Threaded(2)
    BigHOPs.start!(ex, nothing)
    @test BigHOPs.capacity(ex) == 2
    entry1 = BigHOPs.RunEntry(1, (a=1,), Float64[])
    BigHOPs.submit!(ex, entry1, p -> p.a)
    @test BigHOPs.capacity(ex) == 1
    entry2 = BigHOPs.RunEntry(2, (a=2,), Float64[])
    BigHOPs.submit!(ex, entry2, p -> p.a)
    @test BigHOPs.capacity(ex) == 0
    out = BigHOPs.poll(ex)
    append!(out, BigHOPs.poll(ex))
    @test length(out) == 2
    @test BigHOPs.capacity(ex) == 2
    @test Set(e.id for (e, _) in out) == Set([1, 2])
    @test all(e.metadata[:start_time] <= e.metadata[:end_time] for (e, _) in out)
    BigHOPs.shutdown!(ex)

    ex_batch = Threaded(4)
    BigHOPs.start!(ex_batch, nothing)
    entryA = BigHOPs.RunEntry(1, (a=1,), Float64[])
    entryB = BigHOPs.RunEntry(2, (a=2,), Float64[])
    entryC = BigHOPs.RunEntry(3, (a=3,), Float64[])
    put!(ex_batch.results, (entryA, BigHOPs.ObjectiveOutcome(10, nothing), now(), now()))
    put!(ex_batch.results, (entryB, InterruptException(), now(), now()))
    put!(ex_batch.results, (entryC, BigHOPs.ObjectiveOutcome(30, nothing), now(), now()))
    ex_batch.in_flight = 3
    out_batch = BigHOPs.poll(ex_batch)
    @test Set(e.id for (e, _) in out_batch) == Set([1, 3])
    @test BigHOPs.capacity(ex_batch) == 0
    @test_throws InterruptException BigHOPs.poll(ex_batch)
    BigHOPs.shutdown!(ex_batch)

    ex_shutdown = Threaded(4)
    BigHOPs.start!(ex_shutdown, nothing)
    BigHOPs.submit!(ex_shutdown, BigHOPs.RunEntry(1, (a=1,), Float64[]), _ -> (sleep(0.3); 1))
    BigHOPs.shutdown!(ex_shutdown)
    @test isready(ex_shutdown.results)

    failed_task = @async throw(InterruptException())
    try
        wait(failed_task)
    catch
    end
    @test istaskfailed(failed_task)
    ex_task_failure = Threaded(1)
    BigHOPs.start!(ex_task_failure, nothing)
    push!(ex_task_failure.tasks, failed_task)
    @test_throws TaskFailedException BigHOPs.shutdown!(ex_task_failure)

    ho = Hyperoptimizer(p -> p.a^2, (a=Continuous(1, 500),); n=500)
    run!(ho; executor=Threaded(8))
    @test length(ho.runs) == 500
    @test all(e -> e.status == BigHOPs.Completed, ho.runs)
    @test all(e -> e.value == e.params.a^2, ho.runs)

    ho_err = Hyperoptimizer(p -> p.a == 2 ? error("boom") : p.a, (a=Nominal([1, 2, 3]),); n=3)
    ex_err = Threaded(3)
    @test_logs (:warn, r"non-Real") run!(ho_err; executor=ex_err)
    @test length(results(ho_err)) == 2
    @test length(ho_err.runs) == 3
    @test count(e -> e.status == BigHOPs.Failed, ho_err.runs) == 1

    let n_calls = Ref(0)
        global interrupt_after_first_threaded(p) = (n_calls[] += 1; n_calls[] == 1 ? p.a : throw(InterruptException()))
    end
    ho_interrupt = Hyperoptimizer(interrupt_after_first_threaded, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws InterruptException run!(ho_interrupt; executor=Threaded(1))
    @test ho_interrupt.status == BigHOPs.Errored
    @test length(results(ho_interrupt)) == 1
    @test ho_interrupt.n_pending == 0
    @test ho_interrupt.runs[2].status == BigHOPs.Abandoned
    @test_throws ArgumentError run!(ho_interrupt)

    ho_errored = Hyperoptimizer(p -> p.a == 1 ? 1.0 : (sleep(2.0); 2.0), (a=Nominal([1, 2, 3]),);
                                 sampler=BuggySampler(), n=3)
    run_errored_exception = nothing
    try
        run!(ho_errored; executor=Threaded(2))
    catch e
        run_errored_exception = e
    end
    @test run_errored_exception isa ErrorException
    @test run_errored_exception.msg == "sampler bug"
    @test ho_errored.status == BigHOPs.Errored
    @test ho_errored.n_pending == 0
    @test length(ho_errored.runs) == 2
    @test count(e -> e.status == BigHOPs.Completed, ho_errored.runs) == 1
    @test count(e -> e.status == BigHOPs.Abandoned, ho_errored.runs) == 1
    @test_throws ArgumentError run!(ho_errored)
    @test_throws ArgumentError settarget!(ho_errored, 10)
    @test_throws ArgumentError BigHOPs.ask!(ho_errored)
    @test_throws ArgumentError BigHOPs.tell!(ho_errored, ho_errored.runs[2], 42)

    g(p) = (p.a - 7)^2 + (p.b - 3)^2
    ho_exact = Hyperoptimizer(g, (a=Ordinal(0:10), b=Ordinal(0:10)); n=6000)
    run!(ho_exact; executor=Threaded(8))
    @test minimum(ho_exact) == 0
    @test minimizer(ho_exact) == [7, 3]

    @test Threads.nthreads() > 1
    n_slow, delay = 8, 0.2
    ho_slow = Hyperoptimizer(_ -> (sleep(delay); 1.0), (a=Nominal(1:n_slow),); n=n_slow)
    elapsed = @elapsed run!(ho_slow; executor=Threaded(n_slow))
    @test elapsed < n_slow * delay / 2

    calls = Ref(0)
    shared = [[1], [2]]
    mutate_inputs(p) = (calls[] += 1; push!(p.v, 0); calls[] * length(p.v))
    ho_copies = Hyperoptimizer(mutate_inputs, (v=Nominal(shared),); n=4)
    run!(ho_copies; executor=Threaded(2), show_progress=false)
    @test calls[] == 0
    @test shared == [[1], [2]]
    @test all(e -> e.value == 2, ho_copies.runs)

    state = [0]
    ex_copy = Threaded(1)
    BigHOPs.start!(ex_copy, nothing)
    BigHOPs.submit!(ex_copy, BigHOPs.RunEntry(1, (a=1,), Float64[]; pre_artefact=state), Stateful((p; pre_artefact) -> (push!(pre_artefact, 1); (1.0, pre_artefact))))
    BigHOPs.poll(ex_copy)
    BigHOPs.shutdown!(ex_copy)
    @test state == [0]

    ho_wait = Hyperoptimizer(_ -> (sleep(2.0); 1.0), (a=Nominal([1, 2]),); n=100)
    run!(ho_wait; executor=Threaded(2), time_cap=1.0, show_progress=false)
    @test ho_wait.status == BigHOPs.Finished
    @test length(ho_wait.runs) == 2
    @test all(e -> e.status == BigHOPs.Completed, ho_wait.runs)

    ho_cut = Hyperoptimizer(_ -> (sleep(5.0); 1.0), (a=Nominal([1, 2]),); n=100)
    elapsed_cut = @elapsed run!(ho_cut; executor=Threaded(2), time_cap=1.0, interrupt=true, show_progress=false)
    @test elapsed_cut < 4.0
    @test ho_cut.status == BigHOPs.Finished
    @test ho_cut.n_pending == 0
    @test count(e -> e.status == BigHOPs.Abandoned, ho_cut.runs) == 2
end
