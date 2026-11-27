@testset "VirtualBenchmarkExecutor" begin
    @info "Testing VirtualBenchmarkExecutor"

    @test_throws ArgumentError VirtualBenchmarkExecutor(0)

    lookup = Stateful((p; pre_artefact) -> (p.a, (training_time=p.a,)))
    virtual_seconds(e) = (e.metadata[:end_time] - e.metadata[:start_time]).value / 1000

    ex = VirtualBenchmarkExecutor(2)
    BigHOPs.start!(ex, nothing)
    @test BigHOPs.capacity(ex) == 2
    BigHOPs.submit!(ex, BigHOPs.RunEntry(1, (a=1000.0,), Float64[]), lookup)
    BigHOPs.submit!(ex, BigHOPs.RunEntry(2, (a=300.0,), Float64[]), lookup)
    @test BigHOPs.capacity(ex) == 0
    first_out = BigHOPs.poll(ex)
    @test [e.id for (e, _) in first_out] == [2]
    @test BigHOPs.capacity(ex) == 1
    BigHOPs.submit!(ex, BigHOPs.RunEntry(3, (a=500.0,), Float64[]), lookup)
    second_out = BigHOPs.poll(ex)
    third_out = BigHOPs.poll(ex)
    @test [e.id for (e, _) in second_out] == [3]
    @test [e.id for (e, _) in third_out] == [1]
    @test isempty(BigHOPs.poll(ex))
    @test 1000 <= ex.clock < 1010
    @test all(abs(virtual_seconds(e) - e.params.a) <= 0.001 for (e, _) in vcat(first_out, second_out, third_out))
    BigHOPs.shutdown!(ex)

    ex_fail = VirtualBenchmarkExecutor(1)
    BigHOPs.start!(ex_fail, nothing)
    BigHOPs.submit!(ex_fail, BigHOPs.RunEntry(1, (a=1.0,), Float64[]), Stateful((p; pre_artefact) -> error("boom")))
    failed = only(BigHOPs.poll(ex_fail))
    @test failed[2] isa ErrorException
    @test virtual_seconds(failed[1]) == 0

    sleepy = Stateful((p; pre_artefact) -> (sleep(p.s); (0.0, (training_time=0.0,))))
    ex_overhead = VirtualBenchmarkExecutor(1)
    BigHOPs.start!(ex_overhead, nothing)
    BigHOPs.submit!(ex_overhead, BigHOPs.RunEntry(1, (s=0.0,), Float64[]), sleepy)
    BigHOPs.poll(ex_overhead)
    clock_before = ex_overhead.clock
    sleep(0.5)
    BigHOPs.submit!(ex_overhead, BigHOPs.RunEntry(2, (s=1.0,), Float64[]), sleepy)
    BigHOPs.poll(ex_overhead)
    @test 0.5 <= ex_overhead.clock - clock_before < 1.0

    fixed_cost = Stateful((p; pre_artefact) -> (p.a, (training_time=1000.0,)))
    for (workers, expected) in ((1, 20_000), (4, 5_000))
        executor = VirtualBenchmarkExecutor(workers)
        ho = Hyperoptimizer(fixed_cost, (a=Continuous(0, 1),); n=20)
        run!(ho; executor, show_progress=false)
        @test all(e -> e.status == BigHOPs.Completed, ho.runs)
        span = (maximum(e.metadata[:end_time] for e in ho.runs) - executor.origin).value / 1000
        @test expected <= span < expected + 60
    end

    ho_plain = Hyperoptimizer(p -> p.a, (a=Continuous(0, 1),); n=2)
    @test_throws ArgumentError run!(ho_plain; executor=VirtualBenchmarkExecutor(2), show_progress=false)
    @test ho_plain.status == BigHOPs.Errored

    for bad in (-1.0, NaN, Inf, "1")
        ho_bad = Hyperoptimizer(Stateful((p; pre_artefact) -> (p.a, (training_time=bad,))), (a=Continuous(0, 1),); n=1)
        @test_throws ArgumentError run!(ho_bad; executor=VirtualBenchmarkExecutor(1), show_progress=false)
    end
end
