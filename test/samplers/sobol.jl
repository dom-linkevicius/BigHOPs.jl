sobol_bowl(p) = (p.a - 0.3)^2 + (p.b - 0.7)^2

@testset "SobolSampler" begin
    @info "Testing SobolSampler"
    candidates = (a=Continuous(0.0, 1.0), b=Continuous(0.0, 1.0))
    unit_params(ho) = [e.unit_params for e in ho.runs]
    strata_hit(ho, dim) = sort(floor.(Int, [e.unit_params[dim] for e in ho.runs] .* length(ho.runs)))

    ho = Hyperoptimizer(p -> sobol_bowl(p) + (p.c ? 0.0 : 1.0), (candidates..., c=Nominal([true, false])); sampler=SobolSampler(), n=16)
    run!(ho; show_progress=false)
    @test all(dim -> strata_hit(ho, dim) == 0:15, 1:3)
    @test all(h -> 0 <= h.a <= 1 && 0 <= h.b <= 1 && h.c isa Bool, history(ho))
    @test !BigHOPs.exhausted(ho.sampler, ho)
    @test !BigHOPs.blocked(ho.sampler, ho)

    @test_logs (:info, r"target set to 32") settarget!(ho, 32)
    run!(ho; show_progress=false)
    @test all(dim -> strata_hit(ho, dim) == 0:31, 1:3)

    seeded(seed, n) = (s = Hyperoptimizer(sobol_bowl, candidates; sampler=SobolSampler(rng=StableRNG(seed)), n=n); run!(s; show_progress=false); s)
    @test unit_params(seeded(1, 16)) == unit_params(seeded(1, 16))
    @test unit_params(seeded(1, 16)) != unit_params(seeded(2, 16))

    shared = SobolSampler()
    first_use = Hyperoptimizer(sobol_bowl, candidates; sampler=shared, n=8)
    run!(first_use; show_progress=false)
    second_use = Hyperoptimizer(sobol_bowl, candidates; sampler=shared, n=8)
    run!(second_use; show_progress=false)
    @test all(dim -> strata_hit(second_use, dim) == 0:7, 1:2)
    @test unit_params(first_use) != unit_params(second_use)

    ho_log = Hyperoptimizer(p -> p.a, (a=Continuous(-1, 3; transform=exp10),); sampler=SobolSampler(), n=16)
    run!(ho_log; show_progress=false)
    @test all(h -> 0.1 <= h.a <= 1000.0, history(ho_log))
    @test strata_hit(ho_log, 1) == 0:15

    discrete = (a=Continuous(1, 50), b=Ordinal(["low", "medium", "high"]), c=Nominal([1, 2, 3]))
    ho_discrete = Hyperoptimizer(p -> p.a + length(p.b) + p.c, discrete; sampler=SobolSampler(), n=48)
    run!(ho_discrete; show_progress=false)
    @test length(results(ho_discrete)) == 48
    @test all(h -> h.b in ("low", "medium", "high") && h.c in (1, 2, 3), history(ho_discrete))

    sh_objective = Stateful((p; pre_artefact=nothing) -> (sobol_bowl(p) + 1 / p.r, nothing))
    for sh_sampler in (Hyperband(R=9, η=3, r_min=1, inner=SobolSampler()), ASHA(R=9, η=3, r_min=1, inner=SobolSampler()))
        ho_sh = Hyperoptimizer(sh_objective, candidates, sh_sampler)
        run!(ho_sh; show_progress=false)
        @test length(ho_sh.completed) == ho_sh.n
        draws = [e.unit_params for e in ho_sh.runs if e.metadata[:rung] == 1]
        @test sort(floor.(Int, 8 .* first.(draws[1:8]))) == 0:7
    end

    mktempdir() do dir
        path = joinpath(dir, "sobol.jld2")
        ho_half = seeded(3, 8)
        save_hyperoptimizer(ho_half, path)
        reloaded = load_hyperoptimizer(sobol_bowl, path)
        settarget!(reloaded, 16)
        run!(reloaded; show_progress=false)
        @test unit_params(reloaded) == unit_params(seeded(3, 16))
    end
end

@testset "SobolSampler optimization correctness" begin
    @info "Testing SobolSampler optimization correctness"

    ho1 = Hyperoptimizer(p -> (p.x - 37)^2, (x=Continuous(1, 100),); sampler=SobolSampler(), n=100)
    run!(ho1; show_progress=false)
    @test abs(minimizer(ho1)[1] - 37) < 0.5

    target = (33.0, 68.0)
    ho2 = Hyperoptimizer(p -> (p.x - target[1])^2 + (p.y - target[2])^2,
                         (x=Continuous(1, 100), y=Continuous(1, 100));
                         sampler=SobolSampler(), n=100)
    run!(ho2; show_progress=false)
    m = minimizer(ho2)
    @test abs(m[1] - target[1]) < 10
    @test abs(m[2] - target[2]) < 10
end
