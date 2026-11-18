@testset "BOHB constructor validation" begin
    @info "Testing BOHB's ρ/q/N_s/N_min/b_w/min_bandwidth validation and the inherited R/η/r_min validation"

    @test_throws ArgumentError BOHB(R=27, ρ=-0.1)
    @test_throws ArgumentError BOHB(R=27, ρ=1.1)
    @test_throws ArgumentError BOHB(R=27, q=0.0)
    @test_throws ArgumentError BOHB(R=27, q=1.0)
    @test_throws ArgumentError BOHB(R=27, N_s=0)
    @test_throws ArgumentError BOHB(R=27, N_min=0)
    @test_throws ArgumentError BOHB(R=27, b_w=0)
    @test_throws ArgumentError BOHB(R=27, min_bandwidth=0)
    @test_throws ArgumentError BOHB(R=0)
    @test_throws ArgumentError BOHB(R=27, η=1)
    @test_throws ArgumentError BOHB(R=27, r_min=0)
    @test_throws ArgumentError BOHB(R=27, iterations=0)

    BOHB(R=27, ρ=0.0, N_min=5)
    BOHB(R=27, ρ=1.0, q=0.5, N_s=1, b_w=1, min_bandwidth=0.1)
end

@testset "BOHBSampler cannot be used on its own" begin
    @info "Testing that a bare BOHBSampler rejects every Sampler entry point"

    b = BigHOPs.BOHBSampler()
    @test BigHOPs.init(b, (a=Nominal([1]),), 5) === b

    @test_throws ArgumentError BigHOPs.on_tell!(b, BigHOPs.RunEntry[], nothing)
    @test_throws ArgumentError BigHOPs.exhausted(b, nothing)
    @test_throws ArgumentError BigHOPs.blocked(b, nothing)
    @test_throws ArgumentError BigHOPs.create_run_entry(b, nothing, 1, (a=1,), [0.5])
    @test_throws ArgumentError b((a=Nominal([1]),), BigHOPs.RunEntry[])

    ho = Hyperoptimizer(p -> p.a, (a=Nominal([1, 2]),), BOHB(R=9, η=3, r_min=1))
    @test ho.sampler.inner isa BigHOPs.BOHBSampler
end

@testset "BOHB resolves N_min against the number of hyperparameters" begin
    @info "Testing that N_min defaults to the number of hyperparameters plus one and can't be set lower"

    candidates = (a=Continuous(0, 1), b=Nominal(3), c=Ordinal(4))

    s = BigHOPs.init(BOHB(R=9), candidates, 10)
    @test s.inner.N_min == 4
    @test isempty(s.inner.observations)
    @test BigHOPs.init(BOHB(R=9, N_min=6), candidates, 10).inner.N_min == 6
    @test_throws ArgumentError BigHOPs.init(BOHB(R=9, N_min=3), candidates, 10)
end

bohb_completed(id, r, u, v) =
    BigHOPs._with_result(BigHOPs.RunEntry(id, (r=r, a=0.0), u, Dict{Symbol,Any}()), BigHOPs.Completed, v, nothing)

@testset "BOHB files only completed trials, by resource" begin
    @info "Testing that BOHB records each completed trial under its resource level and leaves every other trial out"

    b = BigHOPs.init(BOHB(R=9), (a=Continuous(0, 1),), 10).inner
    runs = [bohb_completed(1, 1, [0.1], 1.0), bohb_completed(2, 3, [0.2], 2.0),
            BigHOPs._with_result(BigHOPs.RunEntry(3, (r=1, a=0.0), [0.3]), BigHOPs.Failed, missing, nothing),
            BigHOPs._with_result(BigHOPs.RunEntry(4, (r=1, a=0.0), [0.4]), BigHOPs.TimedOut, missing, nothing),
            bohb_completed(5, 1, [0.5], 3.0)]
    for e in runs
        BigHOPs._tell_inner!(b, nothing, runs, e)
    end

    @test b.observations == Dict(1 => [1, 5], 3 => [2])
end

@testset "BOHB splits the trials at the largest resource level with enough of them" begin
    @info "Testing BOHB's split into disjoint best and next sets, its threshold of more than d trials in each, and its choice of resource level"

    candidates = (a=Continuous(0, 1), b=Continuous(0, 1))
    b = BigHOPs.init(BOHB(R=27), candidates, 10).inner
    runs = BigHOPs.RunEntry[]
    function tell!(r, v)
        push!(runs, bohb_completed(length(runs) + 1, r, rand(StableRNG(length(runs)), 2), v))
        BigHOPs._tell_inner!(b, nothing, runs, last(runs))
    end
    points(mixture) = [mean(c) for c in components(mixture)]
    unit(ids) = [runs[id].unit_params for id in ids]

    @test !BigHOPs._fittable(b.N_min, b.q, 2, 5)
    @test BigHOPs._fittable(b.N_min, b.q, 2, 6)

    foreach(v -> tell!(1, v), [5.0, 1.0, 4.0, 2.0, 3.0])
    tell!(1, 0.5)
    model = BigHOPs._bohb_model(b, candidates, runs)
    @test points(model.l) == unit([6, 2, 4])
    @test points(model.g) == unit([5, 3, 1])
    @test length(components(model.l′)) == 3

    foreach(v -> tell!(1, v), 6.0:20.0)
    model = BigHOPs._bohb_model(b, candidates, runs)
    @test length(components(model.l)) == 3
    @test length(components(model.g)) == 17
    @test isempty(intersect(points(model.l), points(model.g)))

    foreach(v -> tell!(9, v), 1.0:5.0)
    @test BigHOPs._bohb_model(b, candidates, runs).r == 1
    tell!(9, 6.0)
    model = BigHOPs._bohb_model(b, candidates, runs)
    @test model.r == 9
    @test points(model.l) == unit(22:24)
end

@testset "BOHB bandwidths follow Scott's rule with clipping" begin
    @info "Testing BOHB's bandwidths: Scott's rule, capped at (c-1)/c for Nominal with c the declared levels, and floored at min_bandwidth"

    candidates = (a=Continuous(0, 1), b=Nominal(3), c=Continuous(0, 1), d=Nominal(4))
    b = BigHOPs.init(BOHB(R=9, min_bandwidth=0.01), candidates, 10).inner
    us = [[0.1, 0.1, 0.5, 0.3], [0.5, 0.9, 0.5, 0.4], [0.2, 0.1, 0.5, 0.3], [0.6, 0.9, 0.5, 0.4]]
    runs = [bohb_completed(i, 1, u, 1.0) for (i, u) in enumerate(us)]
    X, h = BigHOPs._kde_data(b, candidates, runs, 1:4)

    @test X[2, :] == [1.0, 3.0, 1.0, 3.0]
    @test X[4, :] == [2.0, 2.0, 2.0, 2.0]
    a = [0.1, 0.5, 0.2, 0.6]
    @test h[1] ≈ 1.06 * sqrt(sum((a .- sum(a) / 4) .^ 2) / 4) * 4^(-1 / 8)
    @test h[2] == 2 / 3
    @test h[3] == 0.01
    @test h[4] == 0.01
end

@testset "BOHB kernels and densities" begin
    @info "Testing BOHB's kernels, Aitchison-Aitken on Nominal and Gaussian elsewhere, and their mixture over the data"

    @test BigHOPs._kernel(Continuous(0, 1), 0.4, 0.1) == Normal(0.4, 0.1)
    @test probs(BigHOPs._kernel(Nominal(3), 2.0, 0.2)) ≈ [0.1, 0.8, 0.1]
    @test probs(BigHOPs._kernel(Nominal(1), 1.0, 0.001)) == [1.0]
    @test BigHOPs._proposal_kernel(Continuous(0, 1), 0.4, 0.1, 3.0) == truncated(Normal(0.4, 3.0 * 0.1), 0.0, 1.0)
    @test probs(BigHOPs._proposal_kernel(Nominal(3), 2.0, 0.6, 3.0)) ≈ [0.2, 0.6, 0.2]

    gauss(m, x) = exp(-((x - m) / 0.1)^2 / 2) / (sqrt(2π) * 0.1)
    mixture = BigHOPs._mixture(BigHOPs._kernel, (Nominal(3), Continuous(0, 1)), [2.0 1.0; 0.4 0.6], [0.2, 0.1])
    @test pdf(mixture, [2.0, 0.5]) ≈ (0.8 * gauss(0.4, 0.5) + 0.1 * gauss(0.6, 0.5)) / 2
    @test pdf(mixture, [3.0, 0.4]) ≈ (0.1 * gauss(0.4, 0.4) + 0.1 * gauss(0.6, 0.4)) / 2
end

@testset "BOHB samples around the good trials" begin
    @info "Testing BOHB's samples from the widened good density: declared Nominal levels and values in [0, 1] elsewhere"

    rng = StableRNG(3)
    candidates = (Nominal(3), Continuous(0, 1), Continuous(0, 1))
    l′ = BigHOPs._mixture((c, x, h) -> BigHOPs._proposal_kernel(c, x, h, 3.0), candidates, [2.0; 0.0; 0.9;;], [2 / 3, 0.5, 0.001])
    xs = [rand(rng, l′) for _ in 1:300]

    @test Set(x[1] for x in xs) == Set([1.0, 2.0, 3.0])
    @test all(x -> 0 <= x[2] <= 1, xs)
    @test all(x -> abs(x[3] - 0.9) < 0.05, xs)
end

@testset "BOHB proposes Nominal levels as bin centres and moves towards the good trials" begin
    @info "Testing that a model-based BOHB draw encodes Nominal levels at their bin centres and lands near the best trials"

    candidates = (a=Nominal(4), b=Continuous(0, 1))
    s = BigHOPs.init(BOHB(R=9, ρ=0.0, rng=StableRNG(9)), candidates, 10)
    grid = range(0.025, 0.975; length=20)
    runs = [bohb_completed(i, 1, [0.125 + 0.25 * (i % 4), u], (u - 0.8)^2) for (i, u) in enumerate(grid)]
    for e in runs
        BigHOPs._tell_inner!(s.inner, s, runs, e)
    end
    bracket = BigHOPs._make_bracket(s, 1, 1)

    for _ in 1:30
        u = BigHOPs._sample_sh_inner(s, candidates, runs, BigHOPs._draw(bracket, bracket.rungs[1]))
        @test u[1] in (0.125, 0.375, 0.625, 0.875)
        @test abs(u[2] - 0.8) < 0.4
    end
end

@testset "BOHB keeps its model until a result could change it" begin
    @info "Testing that BOHB refits at the next draw only after a completed result at or above its model's resource level"

    candidates = (a=Continuous(0, 1),)
    s = BigHOPs.init(BOHB(R=9, ρ=0.0, rng=StableRNG(1)), candidates, 10)
    b = s.inner
    runs = BigHOPs.RunEntry[]
    function tell!(r, v)
        push!(runs, bohb_completed(length(runs) + 1, r, [v / 10], v))
        BigHOPs._tell_inner!(b, s, runs, last(runs))
    end
    bracket = BigHOPs._make_bracket(s, 1, 1)
    draw() = BigHOPs._sample_sh_inner(s, candidates, runs, BigHOPs._draw(bracket, bracket.rungs[1]))

    foreach(v -> tell!(3, v), 1.0:4.0)
    @test b.stale[]
    draw()
    @test !b.stale[]
    model = b.model[]
    @test model.r == 3

    tell!(1, 0.5)
    @test !b.stale[]
    draw()
    @test b.model[] === model

    tell!(3, 5.0)
    @test b.stale[]
    draw()
    @test b.model[] !== model
    @test length(components(b.model[].g)) == 3
end

@testset "BOHB draws at random until a resource level has enough trials" begin
    @info "Testing that BOHB's first 2(d+1) draws are uniform draws from its rng, each after its ρ coin"

    ho = Hyperoptimizer(p -> p.a + p.b, (a=Continuous(0, 1), b=Continuous(0, 1)), BOHB(R=9, η=3, r_min=1, rng=StableRNG(5)))
    run!(ho; show_progress=false)
    rng = StableRNG(5)
    expected = map(1:6) do _
        rand(rng)
        return rand(rng, 2)
    end
    @test [e.unit_params for e in ho.runs[1:6]] == expected
end

@testset "BOHB construction and basic run" begin
    @info "Testing a full BOHB run: schedule shape, resource levels, recorded observations, and Hyperband promotions"

    function toy(p; pre_artefact=nothing)
        return (p.a - 3.0)^2 + (p.b == :x ? 0.0 : 1.0) + 1.0 / p.r, p.r
    end

    ho = Hyperoptimizer(Stateful(toy), (a=Continuous(0, 10), b=Nominal([:x, :y, :z])), BOHB(R=27, η=3, r_min=1, iterations=2))
    run!(ho; show_progress=false)

    @test ho.n == 2 * BigHOPs._total_trials(27, 1, 3)
    @test length(ho.runs) == ho.n
    @test length(ho.completed) == ho.n
    @test ho.status == BigHOPs.Finished
    @test :r ∉ ho.params
    @test all(e -> e.params.r in (1, 3, 9, 27), ho.runs)

    b = ho.sampler.inner
    for r in (1, 3, 9, 27)
        @test b.observations[r] == [e.id for e in ho.runs if e.params.r == r]
    end

    promoted = filter(e -> haskey(e.metadata, :promoted_from), ho.runs)
    @test !isempty(promoted)
    for e in promoted
        parent = ho.runs[e.metadata[:promoted_from]]
        @test e.unit_params == parent.unit_params
        @test e.pre_artefact == parent.post_artefact
    end

    @test_throws ArgumentError settarget!(ho, ho.n + 10)
end
