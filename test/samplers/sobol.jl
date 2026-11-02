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

    sh_objective = Stateful((p; pre_artefact=nothing) -> (sobol_bowl(p) + 1 / p.r, nothing))
    ho_sh = Hyperoptimizer(sh_objective, candidates, Hyperband(R=9, η=3, r_min=1, inner=SobolSampler()))
    run!(ho_sh; show_progress=false)
    @test length(ho_sh.completed) == ho_sh.n
    draws = [e.unit_params for e in ho_sh.runs if e.metadata[:rung] == 1]
    @test sort(floor.(Int, 8 .* first.(draws[1:8]))) == 0:7

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
