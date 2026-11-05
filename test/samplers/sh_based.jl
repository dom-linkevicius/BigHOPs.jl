@testset "Hyperband/ASHA constructor validation" begin
    @info "Testing SuccessiveHalving's R/η/r_min validation"

    @test_throws ArgumentError Hyperband(R=0, η=3, r_min=1)
    @test_throws ArgumentError Hyperband(R=-1, η=3, r_min=1)
    @test_throws ArgumentError Hyperband(R=9, η=1, r_min=1)
    @test_throws ArgumentError Hyperband(R=9, η=0, r_min=1)
    @test_throws ArgumentError Hyperband(R=9, η=-3, r_min=1)
    @test_throws ArgumentError Hyperband(R=9, η=3, r_min=0)
    @test_throws ArgumentError Hyperband(R=9, η=3, r_min=-1)
    @test_throws ArgumentError Hyperband(R=1, η=3, r_min=9)

    Hyperband(R=9, η=2, r_min=1)
    Hyperband(R=9, η=3, r_min=9)
end

@testset "Hyperband/ASHA construction and basic run" begin
    @info "Testing Hyperband/ASHA construction and basic run"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    for (name, sampler) in (("Hyperband", Hyperband(R=81, η=3, r_min=1)), ("ASHA", ASHA(R=81, η=3, r_min=1)))
        ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), sampler)
        run!(ho; show_progress=false)
        @test ho.n == length(ho.runs)
        @test length(ho.completed) == ho.n
        @test ho.status == BigHOPs.Finished

        @test :r ∉ ho.params
        @test length(ho.candidates) == 2
        @test keys(ho.runs[1].params)[1] == :r
        @test all(e -> length(e.unit_params) == 2, ho.runs)
        r_values = [e.params.r for e in ho.runs]
        @test all(r -> r in (1, 3, 9, 27, 81), r_values)

        counts = Dict(r => count(==(r), r_values) for r in unique(r_values))
        @test counts[minimum(r_values)] > counts[maximum(r_values)]

        m = minimizer(ho)
        @test abs(m[2] - 3.0) < 1.0
        @test abs(m[3] - 1.0) < 1.0
    end
end

@testset "Hyperband/ASHA warm-start correctness" begin
    @info "Testing Hyperband/ASHA warm-start (pre_artefact/post_artefact) correctness"

    for sampler in (Hyperband(R=27, η=3, r_min=1), ASHA(R=27, η=3, r_min=1))
        call_id = Ref(0)
        function stateful_obj(p; pre_artefact=nothing)
            call_id[] += 1
            resumed_from = pre_artefact === nothing ? nothing : pre_artefact[1]
            loss = p.a + 0.001 * p.b
            return loss, (call_id[], resumed_from)
        end

        ho = Hyperoptimizer(Stateful(stateful_obj), (a=Nominal([0.0, 1.0, 2.0]), b=Nominal([0.0, 1.0])), sampler)
        run!(ho; show_progress=false)

        n_promoted = 0
        for e in ho.runs
            e.status == BigHOPs.Completed || continue
            if e.pre_artefact !== nothing
                n_promoted += 1
                pre_call_id, _ = e.pre_artefact
                @test pre_call_id !== nothing
                producer = findfirst(x -> x.status == BigHOPs.Completed && x.post_artefact[1] == pre_call_id, ho.runs)
                @test producer !== nothing
                @test producer < e.id

                prev = ho.runs[e.metadata[:promoted_from]]
                @test (e.params.a, e.params.b) == (prev.params.a, prev.params.b)
                @test e.params.r > prev.params.r
                @test e.unit_params == prev.unit_params
            end
        end
        @test n_promoted > 0
    end
end

@testset "Hyperband/ASHA warn on non-Stateful objective" begin
    @info "Testing Hyperband/ASHA warn when the objective can't resume from a previous trial's state"

    for sampler in (Hyperband(R=9, η=3, r_min=1), ASHA(R=9, η=3, r_min=1))
        @test_logs (:warn, r"non-Stateful") match_mode = :any Hyperoptimizer(p -> p.a, (a=Nominal([1]),), sampler)
        @test_logs min_level = Logging.Warn Hyperoptimizer(Stateful((p; pre_artefact=nothing) -> (p.a, nothing)),
                                                            (a=Nominal([1]),), sampler)
        @test_logs min_level = Logging.Warn Hyperoptimizer(nothing, (a=Nominal([1]),), sampler)
    end
end

@testset "Hyperband/ASHA reserved :r, explicit n, and failure handling" begin
    @info "Testing Hyperband/ASHA reserved :r name, explicit-n rejection, and NaN/failure exclusion"

    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (r=Nominal([1]), a=Nominal([1])), Hyperband(R=9, η=3, r_min=1))
    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (r=Nominal([1]), a=Nominal([1])), ASHA(R=9, η=3, r_min=1))

    @test_throws MethodError Hyperoptimizer(p -> p.a, (a=Nominal([1]),), Hyperband(R=9, η=3, r_min=1); n=5)
    @test_throws MethodError Hyperoptimizer(p -> p.a, (a=Nominal([1]),), ASHA(R=9, η=3, r_min=1); n=5)

    n_calls = Ref(0)
    global hb_flaky(p) = (n_calls[] += 1; n_calls[] % 5 == 0 ? NaN : p.a + 1.0 / p.r)
    for sampler in (Hyperband(R=9, η=3, r_min=1), ASHA(R=9, η=3, r_min=1))
        n_calls[] = 0
        ho = Hyperoptimizer(hb_flaky, (a=Nominal([1, 2, 3, 4, 5]),), sampler)
        @test_logs (:warn, r"NaN") match_mode = :any run!(ho; show_progress=false)
        n_failed = count(e -> e.status == BigHOPs.Failed, ho.runs)
        @test n_failed > 0
        @test length(ho.completed) == length(ho.runs) - n_failed
    end
end

@testset "Hyperband/ASHA are FixedPlanSamplers" begin
    @info "Testing Hyperband/ASHA are FixedPlanSamplers (fixed one-pass plan, no settarget!/resume)"

    for sampler in (Hyperband(R=9, η=3, r_min=1), ASHA(R=9, η=3, r_min=1))
        @test sampler isa BigHOPs.FixedPlanSampler
        ho = Hyperoptimizer(p -> p.a, (a=Nominal([1, 2, 3]),), sampler)
        run!(ho; show_progress=false)
        @test ho.status == BigHOPs.Finished
        @test_throws ArgumentError settarget!(ho, ho.n + 10)
    end
end

@testset "Hyperband/ASHA multi-iteration schedules" begin
    @info "Testing Hyperband/ASHA replaying the whole bracket schedule for iterations > 1"

    @test_throws ArgumentError Hyperband(R=9, η=3, r_min=1, iterations=0)
    @test_throws ArgumentError ASHA(R=9, η=3, r_min=1, iterations=-1)

    toy(p; pre_artefact=nothing) = ((p.a - 3.0)^2 + 1.0 / p.r, nothing)
    one_pass = BigHOPs._total_trials(9, 1, 3)

    for sampler in (Hyperband(R=9, η=3, r_min=1, iterations=3), ASHA(R=9, η=3, r_min=1, iterations=3))
        ho = Hyperoptimizer(Stateful(toy), (a=Continuous(0, 10),), sampler)
        run!(ho; show_progress=false)
        @test ho.n == 3 * one_pass
        @test length(ho.runs) == ho.n
        @test ho.status == BigHOPs.Finished

        shapes = [sort([(e.metadata[:bracket], e.metadata[:rung])
                        for e in ho.runs if e.metadata[:iteration] == t]) for t in 1:3]
        @test all(!isempty, shapes)
        @test shapes[2] == shapes[1]
        @test shapes[3] == shapes[1]

        for e in ho.runs
            haskey(e.metadata, :promoted_from) || continue
            @test ho.runs[e.metadata[:promoted_from]].metadata[:iteration] == e.metadata[:iteration]
        end
    end
end

@testset "Hyperband/ASHA under Threaded executor" begin
    @info "Testing Hyperband/ASHA under the Threaded executor (cross-bracket promotion correctness under real concurrency)"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    for sampler in (Hyperband(R=81, η=3, r_min=1), ASHA(R=81, η=3, r_min=1))
        ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), sampler)
        run!(ho; executor=Threaded(8), show_progress=false)
        @test length(ho.runs) == ho.n
        @test length(ho.completed) == ho.n
        @test ho.n_pending == 0
        @test ho.status == BigHOPs.Finished
        @test sort([e.id for e in ho.runs]) == collect(1:ho.n)
    end
end

@testset "Hyperband/ASHA/DEHB start the next bracket while earlier ones wait" begin
    @info "Testing that brackets waiting on results let the next bracket, then the next iteration, start, while earlier brackets keep priority"

    for sampler in (Hyperband(R=9, η=3, r_min=1), ASHA(R=9, η=3, r_min=1), DEHB(R=9, η=3, r_min=1))
        ho = Hyperoptimizer(nothing, (a=Continuous(0, 1),), sampler)
        first_rung = [BigHOPs.ask!(ho) for _ in 1:9]
        @test all(e -> e.metadata[:bracket] == 1 && e.metadata[:rung] == 1, first_rung)
        overlap = BigHOPs.ask!(ho)
        @test overlap.metadata[:bracket] == 2 && overlap.metadata[:rung] == 1
        for (i, e) in enumerate(first_rung)
            BigHOPs.tell!(ho, e, Float64(i))
        end
        promoted = BigHOPs.ask!(ho)
        @test promoted.metadata[:bracket] == 1 && promoted.metadata[:rung] == 2
    end

    for sampler in (Hyperband(R=3, η=3, r_min=1, iterations=2), ASHA(R=3, η=3, r_min=1, iterations=2),
                    DEHB(R=3, η=3, r_min=1, iterations=2))
        ho = Hyperoptimizer(nothing, (a=Continuous(0, 1),), sampler)
        first_pass = [BigHOPs.ask!(ho) for _ in 1:5]
        @test [(e.metadata[:bracket], e.metadata[:rung]) for e in first_pass] == [(1, 1), (1, 1), (1, 1), (2, 1), (2, 1)]
        next_iteration = BigHOPs.ask!(ho)
        @test next_iteration.metadata[:iteration] == 2 && next_iteration.metadata[:bracket] == 1
    end

    ho = Hyperoptimizer(nothing, (a=Continuous(0, 1),), Hyperband(R=3, η=3, r_min=1))
    foreach(_ -> BigHOPs.ask!(ho), 1:5)
    @test BigHOPs.blocked(ho.sampler, ho)
    @test !BigHOPs.exhausted(ho.sampler, ho)
end

@testset "Hyperband/ASHA under DistributedQueue executor" begin
    @info "Testing Hyperband/ASHA under the DistributedQueue executor (cross-bracket promotion correctness under real concurrency)"

    @everywhere using BigHOPs
    @everywhere sh_dq_toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    sh_dq_spawn_worker() = first(addprocs(1))
    function sh_dq_setup_worker(pid)
        Distributed.remotecall_eval(Main, [pid], :(begin
            using BigHOPs
            sh_dq_toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r
        end))
        return nothing
    end

    try
        for sampler in (Hyperband(R=9, η=3, r_min=1), ASHA(R=9, η=3, r_min=1))
            ho = Hyperoptimizer(sh_dq_toy, (a=Continuous(0, 10), b=Continuous(0, 5)), sampler)
            run!(ho; executor=DistributedQueue(3; spawn_worker=sh_dq_spawn_worker, setup_worker=sh_dq_setup_worker), show_progress=false)
            @test length(ho.runs) == ho.n
            @test length(ho.completed) == ho.n
            @test ho.n_pending == 0
            @test ho.status == BigHOPs.Finished
            @test sort([e.id for e in ho.runs]) == collect(1:ho.n)
        end
    finally
        rmprocs(filter(!=(1), workers()))
    end
end

@testset "Hyperband rung-level failure handling (shrinkage/abandonment)" begin
    @info "Testing Hyperband's promote-fewer-than-planned and abandon-on-total-wipeout logic"

    s = Hyperband(R=9, η=3, r_min=1)
    b = BigHOPs._open_next!(s)

    function rung1_entries(bracket, n_total, n_failed)
        runs = BigHOPs.RunEntry[]
        for idx in 1:n_total
            e = BigHOPs.RunEntry(idx, (r=1, a=idx), [(idx - 0.5) / n_total],
                               Dict{Symbol,Any}(:rung => 1, :iteration => 1, :bracket => 1))
            e = idx <= n_failed ? BigHOPs._with_result(e, BigHOPs.Failed, missing, nothing; error=NaN) :
                                   BigHOPs._with_result(e, BigHOPs.Completed, Float64(idx), nothing)
            push!(runs, e)
            push!(bracket.rungs[1].ids, idx)
        end
        return runs
    end

    runs = rung1_entries(b, 9, 7)
    logs, _ = Test.collect_test_logs() do
        BigHOPs.on_tell!(s, runs, runs[end])
    end
    @test count(l -> l.level == Logging.Warn && occursin("promoting fewer than planned", l.message), logs) == 1
    d = BigHOPs._bracket_decision(s, b, runs)
    @test d isa BigHOPs.SHDecision{:promote} && d.rung.rung == 1 && d.promoted_from == 8

    push!(runs, BigHOPs.RunEntry(10, (r=3, a=8), [7.5 / 9], Dict{Symbol,Any}(:rung => 2, :iteration => 1, :bracket => 1)))
    push!(b.rungs[2].ids, 10)
    d = BigHOPs._bracket_decision(s, b, runs)
    @test d isa BigHOPs.SHDecision{:promote} && d.promoted_from == 9
    push!(runs, BigHOPs.RunEntry(11, (r=3, a=9), [8.5 / 9], Dict{Symbol,Any}(:rung => 2, :iteration => 1, :bracket => 1)))
    push!(b.rungs[2].ids, 11)
    @test BigHOPs._bracket_decision(s, b, runs) isa BigHOPs.SHDecision{:wait}

    s2 = Hyperband(R=9, η=3, r_min=1)
    b2 = BigHOPs._open_next!(s2)
    wiped = rung1_entries(b2, 9, 9)
    logs2, _ = Test.collect_test_logs() do
        BigHOPs.on_tell!(s2, wiped, wiped[end])
    end
    @test count(l -> l.level == Logging.Warn && occursin("abandoning it", l.message), logs2) == 1
    @test BigHOPs._bracket_decision(s2, b2, wiped) isa BigHOPs.SHDecision{:exhausted}
    moved = BigHOPs._manage_decide!(s2, wiped)
    @test moved isa BigHOPs.SHDecision{:draw} && moved.bracket.bracket == 2 && moved.rung.rung == 1

    flaky(p) = p.a > 4 ? NaN : Float64(p.a) + 1.0 / p.r
    outcomes = map((Serial(), Threaded(4))) do executor
        ho = Hyperoptimizer(flaky, (a=Nominal(collect(1:20)),), Hyperband(R=27, η=3, r_min=1))
        logs3, _ = Test.collect_test_logs() do
            run!(ho; executor=executor, show_progress=false)
        end
        @test ho.status == BigHOPs.Finished
        n_failed = count(e -> e.status == BigHOPs.Failed, ho.runs)
        @test n_failed > 0
        @test length(ho.runs) < ho.n
        @test length(ho.completed) == length(ho.runs) - n_failed
        n_shrunk = count(l -> l.level == Logging.Warn && occursin("promoting fewer than planned", l.message), logs3)
        @test n_shrunk > 0
        return (n_failed, length(ho.runs), n_shrunk, sort(results(ho)))
    end
    @test outcomes[1] == outcomes[2]
end

@testset "Hyperband/ASHA with LHSampler inner" begin
    @info "Testing Hyperband/ASHA with LHSampler as inner (fresh draws via LHS instead of RandomSampler)"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    for sampler in (Hyperband(R=27, η=3, r_min=1; inner=LHSampler()), ASHA(R=27, η=3, r_min=1; inner=LHSampler()))
        ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), sampler)
        run!(ho; show_progress=false)
        @test ho.n == length(ho.runs)
        @test length(ho.completed) == ho.n
        @test ho.status == BigHOPs.Finished
    end
end

@testset "ASHA rung-level failure handling (bracket-stall/rung-failure warnings)" begin
    @info "Testing ASHA's on_tell! bracket-stalled and rung-completed-with-failure warnings"

    s = ASHA(R=9, η=3, r_min=1)
    b = BigHOPs._open_next!(s)

    function rung1_entries(bracket, n_total, n_failed)
        runs = BigHOPs.RunEntry[]
        for idx in 1:n_total
            e = BigHOPs.RunEntry(idx, (r=1, a=idx), [(idx - 0.5) / n_total],
                               Dict{Symbol,Any}(:rung => 1, :iteration => 1, :bracket => 1))
            e = idx <= n_failed ? BigHOPs._with_result(e, BigHOPs.Failed, missing, nothing; error=NaN) :
                                   BigHOPs._with_result(e, BigHOPs.Completed, Float64(idx), nothing)
            push!(runs, e)
            push!(bracket.rungs[1].ids, idx)
        end
        return runs
    end

    partial = rung1_entries(b, 9, 1)
    logs1, _ = Test.collect_test_logs() do
        BigHOPs.on_tell!(s, partial, partial[end])
    end
    @test count(l -> l.level == Logging.Warn && occursin("completed with at least one failed trial", l.message), logs1) == 1
    @test count(l -> l.level == Logging.Warn && occursin("stalled", l.message), logs1) == 0

    s2 = ASHA(R=9, η=3, r_min=1)
    b2 = BigHOPs._open_next!(s2)
    wiped = rung1_entries(b2, 9, 9)
    logs2, _ = Test.collect_test_logs() do
        BigHOPs.on_tell!(s2, wiped, wiped[end])
    end
    @test count(l -> l.level == Logging.Warn && occursin("bracket 1 of iteration 1 stalled at 9/13", l.message), logs2) == 1
    @test count(l -> l.level == Logging.Warn && occursin("rung 1 of bracket 1 of iteration 1 completed with at least one failed trial", l.message), logs2) == 1
    @test BigHOPs._bracket_decision(s2, b2, wiped) isa BigHOPs.SHDecision{:exhausted}
    moved = BigHOPs._manage_decide!(s2, wiped)
    @test moved isa BigHOPs.SHDecision{:draw} && moved.bracket.bracket == 2 && moved.rung.rung == 1

    flaky(p) = p.a > 4 ? NaN : Float64(p.a) + 1.0 / p.r
    for (label, executor) in (("Serial", Serial()), ("Threaded", Threaded(4)))
        ho = Hyperoptimizer(flaky, (a=Nominal(collect(1:20)),), ASHA(R=27, η=3, r_min=1))
        logs3, _ = Test.collect_test_logs() do
            run!(ho; executor=executor, show_progress=false)
        end
        @test ho.status == BigHOPs.Finished
        n_failed = count(e -> e.status == BigHOPs.Failed, ho.runs)
        @test n_failed > 0
        @test length(ho.runs) < ho.n
        @test length(ho.completed) == length(ho.runs) - n_failed
        @test count(l -> l.level == Logging.Warn && occursin("stalled", l.message), logs3) > 0
        @test count(l -> l.level == Logging.Warn && occursin("completed with at least one failed trial", l.message), logs3) > 0
    end
end

@testset "SuccessiveHalving show reports its schedule" begin
    @info "Testing that displaying a SuccessiveHalving Hyperoptimizer lists resource levels and total trials"

    toy = Stateful((p; pre_artefact=nothing) -> (p.a, nothing))

    hb = Hyperoptimizer(toy, (a=Continuous(0, 1),), Hyperband(R=27, η=3, r_min=1, iterations=2))
    text = sprint(show, hb)
    @test occursin("resource levels: 1, 3, 9, 27", text)
    @test occursin("trials per level: 54, 42, 26, 16", text)
    @test occursin("total trials: 138", text)
    @test occursin("total trials: $(hb.n)", text)

    for sampler in (Hyperband(R=27, η=3, r_min=1, iterations=2), ASHA(R=9, η=3, r_min=1),
                    Hyperband(R=81, η=3, r_min=3), DEHB(R=729, η=3, r_min=9))
        @test sum(BigHOPs._trials_per_level(sampler)) == BigHOPs._total_trials(sampler)
        @test length(BigHOPs._trials_per_level(sampler)) == BigHOPs._smax(sampler.R, sampler.r_min, sampler.η) + 1
    end

    @test occursin("resource levels: 1, 3, 9", sprint(show, Hyperoptimizer(toy, (a=Continuous(0, 1),), ASHA(R=9, η=3, r_min=1))))
    @test occursin("resource levels: 3, 9, 27, 81", sprint(show, Hyperoptimizer(toy, (a=Continuous(0, 1),), Hyperband(R=81, η=3, r_min=3))))
    @test occursin("resource levels: 1, 3, 9", sprint(show, Hyperoptimizer(p -> p.a, (a=Continuous(0, 1),), DEHB(R=9, η=3, r_min=1))))

    plain = sprint(show, Hyperoptimizer(p -> p.a, (a=Continuous(0, 1),); n=5))
    @test !occursin("resource levels", plain)
    @test !occursin("trials per level", plain)
    @test !occursin("total trials", plain)
end
