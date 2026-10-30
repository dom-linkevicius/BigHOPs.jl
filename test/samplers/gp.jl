gp_bowl(p) = (p.x - 0.3)^2 + (p.y - 0.7)^2

@testset "GPSampler" begin
    @info "Testing GPSampler"
    candidates = (x=Continuous(0.0, 1.0), y=Continuous(0.0, 1.0))
    thompson = ThompsonSampling(500; jitter=1e-6)
    gp_sampler(; kwargs...) = GPSampler(GP(Matern52Kernel()); next_point=thompson, noise=1e-6, n_initial=5, kwargs...)

    @test_throws ArgumentError GPSampler(GP(Matern52Kernel()); next_point=thompson, noise=-1, n_initial=5)
    @test_throws ArgumentError GPSampler(GP(Matern52Kernel()); next_point=thompson, noise=1e-6, n_initial=-1)
    @test_throws ArgumentError ThompsonSampling(0; jitter=1e-6)
    @test_throws ArgumentError ThompsonSampling(500; jitter=-1)

    ho = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(rng=StableRNG(1)), n=30)
    run!(ho; show_progress=false)
    rng = StableRNG(1)
    @test [e.unit_params for e in ho.runs[1:5]] == [rand(rng, 2) for _ in 1:5]
    @test !BigHOPs.exhausted(ho.sampler, ho)
    @test !BigHOPs.blocked(ho.sampler, ho)

    ho_random = Hyperoptimizer(gp_bowl, candidates; sampler=RandomSampler(StableRNG(1)), n=30)
    run!(ho_random; show_progress=false)
    @test minimum(ho) < 1e-3
    @test minimum(ho) < minimum(ho_random)

    seen = Tuple{Int,Float64,Float64}[]
    refitted = GP(SqExponentialKernel())
    fit_recorder(X, y) = (push!(seen, (length(X), sum(y), sum(abs2, y))); refitted)
    gp_failing(p) = p.x > 0.5 ? error("simulated failure") : gp_bowl(p)
    ho_fit = Hyperoptimizer(gp_failing, candidates; sampler=gp_sampler(fit=fit_recorder), n=20)
    with_logger(NullLogger()) do
        run!(ho_fit; show_progress=false)
    end
    n_completed = count(e -> e.status == BigHOPs.Completed, ho_fit.runs)
    @test 0 < n_completed < 20
    @test first.(seen) == 1:n_completed
    @test all(abs(s) < 1e-8 for (_, s, _) in seen)
    @test all(isapprox(ss, n - 1) for (n, _, ss) in seen[2:end])
    @test ho_fit.sampler.posterior[].prior === refitted

    ho_prior = Hyperoptimizer(gp_bowl, candidates; sampler=gp_sampler(n_initial=0), n=1)
    @test all(0 .<= ho_prior.sampler(ho_prior.candidates, ho_prior.runs) .<= 1)

    smooth = posterior(GP(SqExponentialKernel())(ColVecs(rand(StableRNG(2), 1, 20)), 1e-4), randn(StableRNG(3), 20))
    @test_throws Exception rand(StableRNG(4), smooth(ColVecs(rand(StableRNG(5), 1, 1000))))
    point = ThompsonSampling(1000; jitter=1e-6)(smooth, 1, StableRNG(4))
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
