@testset "DEHB constructor validation" begin
    @info "Testing DEHB's F/crossover validation and the inherited R/η/r_min validation"

    @test_throws ArgumentError DEHB(R=27, F=0.0)
    @test_throws ArgumentError DEHB(R=27, F=-0.5)
    @test_throws ArgumentError DEHB(R=27, F=1.5)
    @test_throws ArgumentError DEHB(R=27, crossover=-0.1)
    @test_throws ArgumentError DEHB(R=27, crossover=1.1)
    @test_throws ArgumentError DEHB(R=0)
    @test_throws ArgumentError DEHB(R=27, η=1)
    @test_throws ArgumentError DEHB(R=27, r_min=0)
    @test_throws ArgumentError DEHB(R=1, r_min=9)
    @test_throws ArgumentError DEHB(R=27, iterations=0)

    DEHB(R=27, F=1.0, crossover=0.0)
    DEHB(R=27, F=0.5, crossover=1.0)
end

@testset "DEHBSampler cannot be used on its own" begin
    @info "Testing that a bare DEHBSampler rejects every Sampler entry point"

    de = BigHOPs.DEHBSampler()
    @test BigHOPs.init(de, (a=Nominal([1]),), 5) === de

    @test_throws ArgumentError BigHOPs.on_tell!(de, BigHOPs.RunEntry[], nothing)
    @test_throws ArgumentError BigHOPs.exhausted(de, nothing)
    @test_throws ArgumentError BigHOPs.blocked(de, nothing)
    @test_throws ArgumentError BigHOPs.create_run_entry(de, nothing, 1, (a=1,), [0.5])
    @test_throws ArgumentError de((a=Nominal([1]),), BigHOPs.RunEntry[])

    ho = Hyperoptimizer(p -> p.a, (a=Nominal([1, 2]),), DEHB(R=9, η=3, r_min=1))
    @test ho.sampler.inner isa BigHOPs.DEHBSampler
end

@testset "DEHB subpopulations are keyed by resource and seeded at random" begin
    @info "Testing that DEHB fills one subpopulation per resource level with unevaluated random members, sized by the largest rung at that level"

    s = DEHB(R=27, η=3, r_min=1)
    de = BigHOPs._subpopulations!(s, 2)

    @test sort(collect(keys(de.subpops))) == [1, 3, 9, 27]
    @test sort(collect(keys(de.total_dispatched))) == [1, 3, 9, 27]
    @test [length(de.subpops[r]) for r in (1, 3, 9, 27)] == [27, 12, 6, 4]
    @test all(members -> all(m -> length(m) == 2 && all(0 .<= m .<= 1), members), values(de.subpops))
    @test all(f -> all(isinf, f), values(de.fitness))
    @test all(ids -> all(==(0), ids), values(de.ids))
    @test all(==(0), values(de.total_dispatched))

    @test BigHOPs._subpopulations!(s, 2) === de
    @test length(de.subpops) == 4
end

@testset "DEHB rolling slot pointer wraps within a subpopulation" begin
    @info "Testing that DEHB's target pointer walks a subpopulation and resets at its end"

    s = DEHB(R=27, η=3, r_min=1)
    de = BigHOPs._subpopulations!(s, 2)

    @test BigHOPs._next_slot(s, 3) == 1
    de.total_dispatched[3] = 11
    @test BigHOPs._next_slot(s, 3) == 12
    de.total_dispatched[3] = 12
    @test BigHOPs._next_slot(s, 3) == 1
    de.total_dispatched[3] = 13
    @test BigHOPs._next_slot(s, 3) == 2
end

@testset "DEHB mutation parents are the best members of a subpopulation" begin
    @info "Testing that DEHB's mutation pool ranks a subpopulation by fitness, unevaluated members last, and still evolves from a pool smaller than three"

    s = DEHB(R=27, η=3, r_min=1)
    de = BigHOPs._subpopulations!(s, 2)
    de.fitness[1][[5, 2, 9]] = [3.0, 1.0, 2.0]

    @test BigHOPs._mutation_pool(de, 1, 2) == de.subpops[1][[2, 9]]
    @test BigHOPs._mutation_pool(de, 1, 4) == de.subpops[1][[2, 9, 5, 1]]
    @test length(BigHOPs._mutation_pool(de, 27, 1)) == 1

    for _ in 1:20
        trial = BigHOPs._evolve(s, 2, 27, 27, 1)
        @test length(trial) == 2
        @test all(0 .<= trial .<= 1)
    end
end

@testset "DEHB promotes the best unpromoted configurations from the level below" begin
    @info "Testing DEHB's promotion queue: ranked by fitness, skipping configurations already at the higher level, peeked until dispatched"

    s = DEHB(R=9, η=3, r_min=1)
    de = BigHOPs._subpopulations!(s, 1)
    de.fitness[1][1:4] = [3.0, 1.0, 2.0, 4.0]
    de.ids[1][1:4] = [11, 12, 13, 14]
    de.subpops[3][1] = copy(de.subpops[1][2])

    bracket = BigHOPs._make_bracket(s, 1, 1)
    decision = BigHOPs._promote(bracket, bracket.rungs[1], 0)

    @test BigHOPs._promotion_candidate!(s, 1, decision) == 13
    @test de.promotion_queues[3] == [13, 11, 14]
    @test BigHOPs._promotion_candidate!(s, 1, decision) == 13
    popfirst!(de.promotion_queues[3])
    @test BigHOPs._promotion_candidate!(s, 1, decision) == 11

    empty!(de.promotion_queues[3])
    for i in 1:4
        de.subpops[3][i] = copy(de.subpops[1][i])
    end
    @test BigHOPs._promotion_candidate!(s, 1, decision) == 12
    @test isempty(de.promotion_queues[3])
end

@testset "DEHB trial generation mixes mutant and target" begin
    @info "Testing DEHB's mutation/crossover: the forced dimension, the crossover rate, and the range reset"

    target = [0.5, 0.5, 0.5, 0.5]
    parents = [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6], [0.15, 0.25, 0.35, 0.45]]

    no_cross = BigHOPs.DEHBSampler(; F=0.5, crossover=0.0, rng=StableRNG(7))
    for _ in 1:20
        trial = BigHOPs._de_trial(no_cross, target, copy(parents))
        @test length(trial) == length(target)
        @test all(0 .<= trial .<= 1)
        @test count(trial .!= target) == 1
    end

    all_cross = BigHOPs.DEHBSampler(; F=0.5, crossover=1.0, rng=StableRNG(7))
    for _ in 1:20
        trial = BigHOPs._de_trial(all_cross, target, copy(parents))
        @test all(0 .<= trial .<= 1)
        @test count(trial .!= target) == length(target)
    end

    wide = BigHOPs.DEHBSampler(; F=1.0, crossover=1.0, rng=StableRNG(3))
    for _ in 1:50
        @test all(0 .<= BigHOPs._de_trial(wide, [0.99, 0.01], [[0.99, 0.01], [0.01, 0.99], [0.5, 0.5]]) .<= 1)
    end

    others = parents
    possible = [a .+ 0.5 .* (b .- c) for a in others, b in others, c in others if a != b && b != c && a != c]
    de = BigHOPs.DEHBSampler(; F=0.5, rng=StableRNG(11))
    for _ in 1:20
        @test BigHOPs._mutant(de, target, vcat([target], others)) in possible
    end
    @test length(BigHOPs._mutant(de, target, [target, others[1]])) == length(target)
end

@testset "DEHB warns only on a Stateful objective" begin
    @info "Testing that DEHB warns about pre_artefact for Stateful objectives and stays quiet otherwise"

    sampler() = DEHB(R=9, η=3, r_min=1)
    stateful = Stateful((p; pre_artefact=nothing) -> (p.a, nothing))

    @test_logs (:warn, r"promotions in the first iteration") match_mode = :any Hyperoptimizer(stateful, (a=Nominal([1]),), sampler())
    @test_logs min_level = Logging.Warn Hyperoptimizer(p -> p.a, (a=Nominal([1]),), sampler())
    @test_logs min_level = Logging.Warn Hyperoptimizer(nothing, (a=Nominal([1]),), sampler())
end

@testset "DEHB construction and basic run" begin
    @info "Testing a full DEHB run: schedule shape, resource levels, and per-trial bookkeeping"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    s = DEHB(R=27, η=3, r_min=1)
    ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), s)
    run!(ho; show_progress=false)

    @test ho.n == BigHOPs._total_trials(27, 1, 3)
    @test length(ho.runs) == ho.n
    @test length(ho.completed) == ho.n
    @test ho.status == BigHOPs.Finished

    @test :r ∉ ho.params
    @test all(e -> e.params.r in (1, 3, 9, 27), ho.runs)
    @test all(e -> length(e.unit_params) == 2, ho.runs)
    @test all(e -> haskey(e.metadata, :slot), ho.runs)

    de = ho.sampler.inner
    for r in (1, 3, 9, 27)
        @test de.total_dispatched[r] == count(e -> e.params.r == r, ho.runs)
        @test all(id -> 1 <= id <= ho.n, filter(!=(0), de.ids[r]))
    end

    @test_throws ArgumentError settarget!(ho, ho.n + 10)
    @test ho.sampler isa BigHOPs.FixedPlanSampler
end

@testset "DEHB keeps the best trial in each subpopulation slot" begin
    @info "Testing DEHB's selection step: a slot is only overwritten by a trial scoring at least as well"

    toy(p) = (p.a - 3.0)^2 + 1.0 / p.r

    ho = Hyperoptimizer(toy, (a=Continuous(0, 10),), DEHB(R=27, η=3, r_min=1))
    run!(ho; show_progress=false)

    de = ho.sampler.inner
    n_checked = 0
    for (r, ids) in de.ids, (slot, id) in enumerate(ids)
        id == 0 && continue
        entry = ho.runs[id]
        @test entry.status == BigHOPs.Completed
        @test entry.params.r == r
        @test entry.metadata[:slot] == slot
        @test de.subpops[r][slot] == entry.unit_params
        @test de.fitness[r][slot] == entry.value

        rivals = [e for e in ho.runs
                  if e.status == BigHOPs.Completed && e.params.r == r && e.metadata[:slot] == slot]
        @test entry.value == minimum(e.value for e in rivals)
        n_checked += 1
    end
    @test n_checked > 0
end

@testset "DEHB selection replaces a slot on ties" begin
    @info "Testing that an equally good trial replaces a slot's occupant, so each slot holds its latest trial under a constant objective"

    ho = Hyperoptimizer(p -> 1.0, (a=Continuous(0, 1),), DEHB(R=9, η=3, r_min=1))
    run!(ho; show_progress=false)

    de = ho.sampler.inner
    for (r, ids) in de.ids, (slot, id) in enumerate(ids)
        id == 0 && continue
        @test id == maximum(e.id for e in ho.runs if e.params.r == r && e.metadata[:slot] == slot)
    end
end

@testset "DEHB promotes unchanged only in the first iteration" begin
    @info "Testing that DEHB copies configurations from the level below into later rungs throughout iteration 1 and evolves every trial afterwards"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), DEHB(R=27, η=3, r_min=1, iterations=2))
    run!(ho; show_progress=false)

    n_copied, n_evolved = 0, 0
    for e in ho.runs
        if e.metadata[:rung] > 1 && e.metadata[:iteration] == 1
            @test haskey(e.metadata, :promoted_from)
            parent = ho.runs[e.metadata[:promoted_from]]
            @test e.unit_params == parent.unit_params
            @test parent.status == BigHOPs.Completed
            @test e.params.r == 3 * parent.params.r
            n_copied += 1
        else
            @test !haskey(e.metadata, :promoted_from)
            n_evolved += 1
        end
    end
    @test n_copied > 0
    @test any(e -> e.metadata[:bracket] > 1 && haskey(e.metadata, :promoted_from), ho.runs)
    @test any(e -> e.metadata[:rung] > 1 && e.metadata[:iteration] == 2, ho.runs)
    @test n_evolved > 0
end

@testset "DEHB carries pre_artefact only through first-iteration promotions" begin
    @info "Testing that promotions in iteration 1 resume from the promoted trial's state and nothing else does"

    call_id = Ref(0)
    function stateful_obj(p; pre_artefact=nothing)
        call_id[] += 1
        return (p.a - 3.0)^2 + 1.0 / p.r, (call_id[], pre_artefact === nothing ? nothing : pre_artefact[1])
    end

    ho = @test_logs (:warn, r"promotions in the first iteration") match_mode = :any Hyperoptimizer(
        Stateful(stateful_obj), (a=Continuous(0, 10),), DEHB(R=27, η=3, r_min=1, iterations=2))
    run!(ho; show_progress=false)

    resumed = filter(e -> e.pre_artefact !== nothing, ho.runs)
    @test !isempty(resumed)
    @test all(e -> e.metadata[:iteration] == 1 && e.metadata[:rung] > 1, resumed)
    @test all(e -> e.pre_artefact == ho.runs[e.metadata[:promoted_from]].post_artefact, resumed)
    @test all(e -> e.post_artefact !== nothing, ho.runs)
end

@testset "DEHB multi-iteration schedules" begin
    @info "Testing that DEHB replays the bracket schedule and stops copying after the first iteration"

    toy(p) = (p.a - 3.0)^2 + 1.0 / p.r
    one_pass = BigHOPs._total_trials(9, 1, 3)

    ho = Hyperoptimizer(toy, (a=Continuous(0, 10),), DEHB(R=9, η=3, r_min=1, iterations=3))
    run!(ho; show_progress=false)

    @test ho.n == 3 * one_pass
    @test length(ho.runs) == ho.n
    @test ho.status == BigHOPs.Finished

    shapes = [sort([(e.metadata[:bracket], e.metadata[:rung])
                    for e in ho.runs if e.metadata[:iteration] == t]) for t in 1:3]
    @test shapes[2] == shapes[1]
    @test shapes[3] == shapes[1]

    @test all(e -> e.metadata[:iteration] == 1, filter(e -> haskey(e.metadata, :promoted_from), ho.runs))
end

@testset "DEHB under the Threaded executor" begin
    @info "Testing DEHB's subpopulation bookkeeping under real concurrency"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), DEHB(R=27, η=3, r_min=1))
    run!(ho; executor=Threaded(4), show_progress=false)

    @test length(ho.runs) == ho.n
    @test length(ho.completed) == ho.n
    @test ho.n_pending == 0
    @test ho.status == BigHOPs.Finished
    @test sort([e.id for e in ho.runs]) == collect(1:ho.n)

    de = ho.sampler.inner
    for r in keys(de.subpops)
        @test de.total_dispatched[r] == count(e -> e.params.r == r, ho.runs)
    end
end
