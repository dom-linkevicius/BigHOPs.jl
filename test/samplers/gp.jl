gp_bowl(p) = (p.x - 0.3)^2 + (p.y - 0.7)^2

@testset "GPSampler" begin
    @info "Testing GPSampler"
    candidates = (x=Continuous(0.0, 1.0), y=Continuous(0.0, 1.0))
    thompson = ThompsonSampling(9; jitter=1e-6)
    gp_sampler(; kwargs...) = GPSampler(GP(Matern52Kernel()); next_point=thompson, noise=1e-6, n_initial=5, kwargs...)

    @test_throws ArgumentError GPSampler(GP(Matern52Kernel()); next_point=thompson, noise=-1, n_initial=5)
    @test_throws ArgumentError GPSampler(GP(Matern52Kernel()); next_point=thompson, noise=1e-6, n_initial=-1)
    @test_throws MethodError GPSampler(GP(Matern52Kernel())(ColVecs(rand(2, 3)), 1e-6); next_point=thompson, noise=1e-6, n_initial=5)
    @test_throws ArgumentError ThompsonSampling(0; jitter=1e-6)
    @test_throws ArgumentError ThompsonSampling(9; jitter=-1)

    ho = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(rng=StableRNG(1)), n=30)
    run!(ho; show_progress=false)
    rng = StableRNG(1)
    @test [e.unit_params for e in ho.runs[1:5]] == [rand(rng, 2) for _ in 1:5]
    @test !BigHOPs.exhausted(ho.sampler, ho)
    @test !BigHOPs.blocked(ho.sampler, ho)

    ho_block = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(n_initial=2, strict_initialization=false), n=5)
    first_trial = BigHOPs.ask!(ho_block)
    @test !BigHOPs.blocked(ho_block.sampler, ho_block)
    second_trial = BigHOPs.ask!(ho_block)
    @test BigHOPs.blocked(ho_block.sampler, ho_block)
    BigHOPs.tell!(ho_block, first_trial, 1.0)
    @test BigHOPs.blocked(ho_block.sampler, ho_block)
    with_logger(NullLogger()) do
        BigHOPs.tell!(ho_block, second_trial, ErrorException("simulated failure"))
    end
    @test !BigHOPs.blocked(ho_block.sampler, ho_block)
    replacement = @test_logs (:warn, r"only 1 of the 2 initial trials have completed") BigHOPs.ask!(ho_block)
    @test BigHOPs.blocked(ho_block.sampler, ho_block)
    BigHOPs.tell!(ho_block, replacement, 2.0)
    @test !BigHOPs.blocked(ho_block.sampler, ho_block)
    @test_logs BigHOPs.ask!(ho_block)
    @test ho_block.sampler.posterior[] !== nothing

    ho_strict = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(n_initial=2, strict_initialization=true), n=5)
    strict_trials = [BigHOPs.ask!(ho_strict) for _ in 1:2]
    with_logger(NullLogger()) do
        BigHOPs.tell!(ho_strict, strict_trials[1], ErrorException("simulated failure"))
    end
    @test BigHOPs.blocked(ho_strict.sampler, ho_strict)
    BigHOPs.tell!(ho_strict, strict_trials[2], 1.0)
    @test !BigHOPs.blocked(ho_strict.sampler, ho_strict)
    @test_logs (:warn, r"only 1 of the 2 initial trials completed; strict_initialization") BigHOPs.ask!(ho_strict)
    @test_logs BigHOPs.ask!(ho_strict)
    @test ho_strict.sampler.posterior[] !== nothing
    ho_strict_ok = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(strict_initialization=true), n=8)
    @test_logs run!(ho_strict_ok; show_progress=false)
    @test length(ho_strict_ok.completed) == 8

    ho_random = Hyperoptimizer(gp_bowl, candidates; sampler=RandomSampler(StableRNG(1)), n=30)
    run!(ho_random; show_progress=false)
    @test minimum(ho) < 1e-3
    @test minimum(ho) < minimum(ho_random)

    seen = Tuple{Int,Float64,Float64}[]
    refitted = GP(SqExponentialKernel())
    fit_recorder(X, y) = (push!(seen, (length(X), sum(y), sum(abs2, y))); refitted)
    gp_failing(p) = p.x > 0.5 ? error("simulated failure") : gp_bowl(p)
    ho_fit = Hyperoptimizer(gp_failing, candidates; sampler=gp_sampler(fit_hyperparams=fit_recorder), n=20)
    with_logger(NullLogger()) do
        run!(ho_fit; show_progress=false)
    end
    n_completed = count(e -> e.status == BigHOPs.Completed, ho_fit.runs)
    @test 0 < n_completed < 20
    @test first.(seen) == 1:n_completed
    @test all(abs(s) < 1e-8 for (_, s, _) in seen)
    @test all(isapprox(ss, n - 1) for (n, _, ss) in seen[2:end])
    @test ho_fit.sampler.posterior[].prior === refitted
    @test ho_fit.sampler.noise[] == 1e-6

    noise_sampler = gp_sampler(fit_hyperparams=(X, y) -> (refitted, 0.25))
    ho_noise = Hyperoptimizer(gp_bowl, candidates; sampler=noise_sampler, n=8)
    @test_logs (:warn, r"noise=1.0e-6 is replaced") run!(ho_noise; show_progress=false)
    @test ho_noise.sampler.noise[] == 0.25
    @test noise_sampler.noise[] == 1e-6

    fitted_noise(fit_hyperparams) = GPSampler(GP(Matern52Kernel()); next_point=thompson, noise=nothing, n_initial=5, fit_hyperparams)
    @test_throws ArgumentError GPSampler(GP(Matern52Kernel()); next_point=thompson, noise=nothing, n_initial=5)
    ho_nothing = Hyperoptimizer(gp_bowl, candidates; sampler=fitted_noise((X, y) -> (refitted, 0.25)), n=8)
    @test_logs run!(ho_nothing; show_progress=false)
    @test ho_nothing.sampler.noise[] == 0.25
    for bad_fit in ((X, y) -> refitted, (X, y) -> (refitted, -1.0), (X, y) -> (refitted, 0.25, 1))
        @test_throws ArgumentError run!(Hyperoptimizer(gp_bowl, candidates; sampler=fitted_noise(bad_fit), n=2); show_progress=false)
    end

    raw = Vector{Float64}[]
    ho_raw = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(standardize=nothing, fit_hyperparams=(X, y) -> (push!(raw, copy(y)); refitted)), n=8)
    run!(ho_raw; show_progress=false)
    @test last(raw) == [e.value for e in ho_raw.runs]
    ho_negated = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(standardize=y -> -y, fit_hyperparams=(X, y) -> (push!(raw, copy(y)); refitted)), n=8)
    run!(ho_negated; show_progress=false)
    @test last(raw) == [-e.value for e in ho_negated.runs]

    mixed = (a=Continuous(-1, 3; transform=exp10), b=Ordinal([1, 10, 100]), c=Nominal([true, false]), d=Nominal([sin, exp, identity]))
    ho_mixed = Hyperoptimizer(p -> p.d(p.b / 100) + log10(p.a)^2 + (p.c ? 0.0 : 1.0), mixed; sampler=gp_sampler(), n=20)
    run!(ho_mixed; show_progress=false)
    @test length(ho_mixed.completed) == 20
    @test all(h -> 0.1 <= h.a <= 1000 && h.b in (1, 10, 100) && h.c isa Bool && h.d in (sin, exp, identity), history(ho_mixed))
    show(devnull, ho_mixed)
    printmin(devnull, ho_mixed)

    ho_prior = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(n_initial=0), n=1)
    @test all(0 .<= ho_prior.sampler(ho_prior.candidates, ho_prior.runs) .<= 1)

    smooth = posterior(GP(SqExponentialKernel())(ColVecs(rand(StableRNG(2), 1, 20)), 1e-4), randn(StableRNG(3), 20))
    @test_throws Exception rand(StableRNG(4), smooth(ColVecs(rand(StableRNG(5), 1, 1000))))
    point = ThompsonSampling(10; jitter=1e-6)(smooth, 1, StableRNG(4))
    @test length(point) == 1 && 0 <= point[1] <= 1

    mktempdir() do dir
        path = joinpath(dir, "gp.jld2")
        save_hyperoptimizer(ho, path)
        reloaded = load_hyperoptimizer(gp_bowl, path)
        @test reloaded.sampler isa GPSampler
        settarget!(reloaded, 35)
        run!(reloaded; show_progress=false)
        @test length(reloaded.runs) == 35
        @test count(e -> e.status == BigHOPs.Completed, reloaded.runs) == 35
    end
end

@testset "GPSampler under the Threaded executor" begin
    @info "Testing GPSampler under the Threaded executor"
    seen = Int[]
    sampler = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(9; jitter=1e-6), noise=1e-6, n_initial=5,
                        fit_hyperparams=(X, y) -> (push!(seen, length(X)); GP(Matern52Kernel())))
    ho = Hyperoptimizer(gp_bowl, (x=Continuous(0.0, 1.0), y=Continuous(0.0, 1.0)); sampler=sampler, n=30)
    run!(ho; executor=Threaded(8), show_progress=false)
    @test length(ho.completed) == 30
    @test ho.n_pending == 0
    @test ho.status == BigHOPs.Finished
    @test sort([e.id for e in ho.runs]) == collect(1:30)
    @test seen == 1:30

    resolved_when_asked = Int[]
    ho_ref = Ref{Any}()
    function recording_next_point(post, d, rng)
        push!(resolved_when_asked, count(e -> e.status !== BigHOPs.Pending, view(ho_ref[].runs, 1:5)))
        return rand(rng, d)
    end
    slow_bowl(p) = (sleep(0.05 * rand()); gp_bowl(p))
    ho_ref[] = Hyperoptimizer(slow_bowl, (x=Continuous(0.0, 1.0), y=Continuous(0.0, 1.0));
                              sampler=GPSampler(GP(Matern52Kernel()); next_point=recording_next_point, noise=1e-6, n_initial=5), n=20)
    run!(ho_ref[]; executor=Threaded(8), show_progress=false)
    @test length(resolved_when_asked) == 15
    @test all(==(5), resolved_when_asked)
end

@testset "GPSampler under the DistributedQueue executor" begin
    @info "Testing GPSampler under the DistributedQueue executor"

    @everywhere using BigHOPs
    @everywhere gp_dq_bowl(p) = (p.x - 0.3)^2 + (p.y - 0.7)^2

    gp_dq_spawn_worker() = first(addprocs(1))
    function gp_dq_setup_worker(pid)
        Distributed.remotecall_eval(Main, [pid], :(begin
            using BigHOPs
            gp_dq_bowl(p) = (p.x - 0.3)^2 + (p.y - 0.7)^2
        end))
        return nothing
    end

    try
        seen = Int[]
        sampler = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(9; jitter=1e-6), noise=1e-6, n_initial=3,
                            fit_hyperparams=(X, y) -> (push!(seen, length(X)); GP(Matern52Kernel())))
        ho = Hyperoptimizer(gp_dq_bowl, (x=Continuous(0.0, 1.0), y=Continuous(0.0, 1.0)); sampler=sampler, n=8)
        run!(ho; executor=DistributedQueue(3; spawn_worker=gp_dq_spawn_worker, setup_worker=gp_dq_setup_worker), show_progress=false)
        @test length(ho.completed) == 8
        @test ho.n_pending == 0
        @test ho.status == BigHOPs.Finished
        @test sort([e.id for e in ho.runs]) == collect(1:8)
        @test seen == 1:8
    finally
        rmprocs(filter(!=(1), workers()))
    end
end
