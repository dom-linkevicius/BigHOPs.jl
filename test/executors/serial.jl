@testset "Serial executor" begin
    @info "Testing Serial executor"

    ex = Serial()
    BigHOPs.start!(ex, nothing)
    @test BigHOPs.capacity(ex) == 1
    entry = BigHOPs.RunEntry(1, (a=1,), Float64[])
    BigHOPs.submit!(ex, entry, p -> p.a)
    @test BigHOPs.capacity(ex) == 0
    out = BigHOPs.poll(ex)
    @test length(out) == 1
    @test out[1][1].id == 1
    @test out[1][1].metadata[:start_time] <= out[1][1].metadata[:end_time]
    @test BigHOPs.capacity(ex) == 1
    BigHOPs.shutdown!(ex)

    let n_calls = Ref(0)
        global interrupt_after_first(p) = (n_calls[] += 1; n_calls[] == 1 ? p.a : throw(InterruptException()))
    end
    ho_interrupt = Hyperoptimizer(interrupt_after_first, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws InterruptException run!(ho_interrupt)
    @test ho_interrupt.status == BigHOPs.Errored
    @test length(results(ho_interrupt)) == 1
    @test ho_interrupt.n_pending == 0
    @test ho_interrupt.runs[2].status == BigHOPs.Abandoned
    @test_throws ArgumentError run!(ho_interrupt)
    @test_throws ArgumentError settarget!(ho_interrupt, 10)

    @test_throws ArgumentError BigHOPs.ask!(ho_interrupt)
    @test_throws ArgumentError BigHOPs.tell!(ho_interrupt, ho_interrupt.runs[2], 42)

    g(p) = (p.a - 7)^2 + (p.b - 3)^2
    ho_exact = Hyperoptimizer(g, (a=Ordinal(0:10), b=Ordinal(0:10)); n=6000)
    run!(ho_exact; executor=Serial())
    @test minimum(ho_exact) == 0
    @test minimizer(ho_exact) == [7, 3]

    calls = Ref(0)
    shared = [[1], [2]]
    mutate_inputs(p) = (calls[] += 1; push!(p.v, 0); calls[] * length(p.v))
    ho_copies = Hyperoptimizer(mutate_inputs, (v=Nominal(shared),); n=4)
    run!(ho_copies; executor=Serial(), show_progress=false)
    @test calls[] == 0
    @test shared == [[1], [2]]
    @test all(e -> e.value == 2, ho_copies.runs)

    state = [0]
    ex_copy = Serial()
    BigHOPs.submit!(ex_copy, BigHOPs.RunEntry(1, (a=1,), Float64[]; pre_artefact=state), Stateful((p; pre_artefact) -> (push!(pre_artefact, 1); (1.0, pre_artefact))))
    BigHOPs.poll(ex_copy)
    @test state == [0]

    ho_capped = Hyperoptimizer(_ -> (sleep(0.2); 1.0), (a=Nominal([1, 2]),); n=100)
    @test_logs (:warn, r"Serial") run!(ho_capped; executor=Serial(), time_cap=1.0, interrupt=true, show_progress=false)
    @test ho_capped.status == BigHOPs.Finished
    @test 1 <= length(ho_capped.runs) < 100
    @test all(e -> e.status == BigHOPs.Completed, ho_capped.runs)
    @test_throws ArgumentError run!(ho_capped; time_cap=0)
end
