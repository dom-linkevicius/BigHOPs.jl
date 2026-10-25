@testset "Failure handling" begin
    @info "Testing NaN and exception failure handling"

    let n_calls = Ref(0)
        global nan_after_first(p) = (n_calls[] += 1; n_calls[] == 1 ? p.a * p.b : NaN)
    end
    ho_nan = Hyperoptimizer(nan_after_first, (a=Nominal([20]), b=Nominal([1])); n=2)
    @test_logs (:warn, r"NaN") run!(ho_nan)
    @test minimum(ho_nan) == 20
    @test minimizer(ho_nan) == [20, 1]
    @test length(ho_nan.runs) == 2
    @test length(results(ho_nan)) == 1
    @test !any(isnan, results(ho_nan))
    @test ho_nan.runs[2].status == BigHOPs.Failed
    @test ho_nan.runs[2].value === missing
    @test isnan(ho_nan.runs[2].error)
    @test ho_nan.runs[1].error === nothing

    let n_calls = Ref(0)
        global throws_after_first(p) = (n_calls[] += 1; n_calls[] == 1 ? p.a * p.b : error("boom"))
    end
    ho_err = Hyperoptimizer(throws_after_first, (a=Nominal([20]), b=Nominal([1])); n=2)
    @test_logs (:warn, r"non-Real") run!(ho_err)
    @test minimum(ho_err) == 20
    @test minimizer(ho_err) == [20, 1]
    @test length(ho_err.runs) == 2
    @test length(results(ho_err)) == 1
    @test ho_err.runs[2].status == BigHOPs.Failed
    @test ho_err.runs[2].value === missing
    @test ho_err.runs[2].error isa ErrorException && ho_err.runs[2].error.msg == "boom"
    @test ho_err.runs[1].error === nothing

    ho_allnan = Hyperoptimizer(p -> NaN, (a=Nominal([1]),); n=1)
    @test_logs (:warn, r"NaN") run!(ho_allnan)
    @test length(results(ho_allnan)) == 0
    @test_throws ErrorException minimum(ho_allnan)
    @test_throws ErrorException minimizer(ho_allnan)

    let n_calls = Ref(0)
        global nan_with_distinctive_params(p) = (n_calls[] += 1; n_calls[] == 1 ? p.a + p.b : NaN)
    end
    ho_params = Hyperoptimizer(nan_with_distinctive_params, (a=Nominal([777]), b=Nominal([888])); n=2)
    io = IOBuffer()
    Logging.with_logger(Logging.ConsoleLogger(io)) do
        run!(ho_params)
    end
    log_text = String(take!(io))
    @test occursin("777", log_text)
    @test occursin("888", log_text)
    @test occursin("a = 777", log_text)
    @test occursin("b = 888", log_text)
end

@testset "TimedOut trials" begin
    @info "Testing that a TrialTimeout outcome marks a trial TimedOut, and how samplers and checkpoints treat it"

    entry = BigHOPs.RunEntry(1, (a=1,), [0.5])
    told = @test_logs (:warn, r"max_single_runtime") BigHOPs.finalize_entry(entry, BigHOPs.TrialTimeout(2.5))
    @test told.status == BigHOPs.TimedOut
    @test told.value === missing
    @test told.error == BigHOPs.TrialTimeout(2.5)

    ho = Hyperoptimizer(nothing, (a=Continuous(0, 1),), Hyperband(R=3, η=3, r_min=1))
    first_rung = [BigHOPs.ask!(ho) for _ in 1:3]
    @test all(e -> e.params.r == 1, first_rung)
    BigHOPs.tell!(ho, first_rung[1], 0.3)
    BigHOPs.tell!(ho, first_rung[2], 0.1)
    @test_logs (:warn, r"max_single_runtime") BigHOPs.tell!(ho, first_rung[3], BigHOPs.TrialTimeout(2.5))
    @test ho.n_pending == 0
    @test BigHOPs._rung_has_failure(ho.runs, first(ho.sampler.active_brackets).rungs[1])

    promoted = BigHOPs.ask!(ho)
    @test promoted.params.r == 3
    @test promoted.metadata[:promoted_from] == first_rung[2].id
    BigHOPs.tell!(ho, promoted, 0.05)

    mktempdir() do dir
        path = joinpath(dir, "timed_out.jld2")
        save_hyperoptimizer(ho, path)
        loaded = load_hyperoptimizer(nothing, path)
        @test loaded.runs[first_rung[3].id].status == BigHOPs.TimedOut
        @test loaded.runs[first_rung[3].id].error == BigHOPs.TrialTimeout(2.5)
    end
end
