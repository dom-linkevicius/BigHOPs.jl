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

@testset "DEHB subpopulations are keyed by resource" begin
    @info "Testing that DEHB allocates one subpopulation per resource level, sized by that bracket's first rung"

    s = DEHB(R=27, η=3, r_min=1)
    de = BigHOPs._subpopulations!(s)

    @test sort(collect(keys(de.subpops))) == [1, 3, 9, 27]
    @test [length(de.subpops[r]) for r in (1, 3, 9, 27)] == [27, 12, 6, 4]
    @test all(slots -> all(==(0), slots), values(de.subpops))

    @test BigHOPs._subpopulations!(s) === de
    @test length(de.subpops) == 4
end

@testset "DEHB rolling slot pointer wraps within a subpopulation" begin
    @info "Testing that DEHB's target pointer counts the trials already at a resource level and wraps at the subpopulation's end"

    s = DEHB(R=27, η=3, r_min=1)
    at(r, n) = [BigHOPs.RunEntry(i, (r=r, a=0.0), [0.5]) for i in 1:n]

    @test BigHOPs._next_slot(s, BigHOPs.RunEntry[], 3) == 1
    @test BigHOPs._next_slot(s, at(3, 11), 3) == 12
    @test BigHOPs._next_slot(s, at(3, 12), 3) == 1
    @test BigHOPs._next_slot(s, [at(3, 13); at(1, 5)], 3) == 2
end

@testset "DEHB mutation parents are the best members of a subpopulation" begin
    @info "Testing that DEHB's mutation pool ranks a subpopulation's occupants by value, pads it with random members up to the rung size, and still evolves from a pool smaller than three"

    completed(id, u, v) = BigHOPs._with_result(BigHOPs.RunEntry(id, (r=1, a=0.0), u, Dict{Symbol,Any}()), BigHOPs.Completed, v, nothing)
    runs = [completed(1, [0.1, 0.1], 3.0), completed(2, [0.2, 0.2], 1.0), completed(3, [0.3, 0.3], 2.0)]
    s = DEHB(R=27, η=3, r_min=1)
    de = BigHOPs._subpopulations!(s)
    de.subpops[1][[5, 2, 9]] = [1, 2, 3]

    @test BigHOPs._members(s, runs, 1, 2, 2) == [[0.2, 0.2], [0.3, 0.3]]
    padded = BigHOPs._members(s, runs, 1, 4, 2)
    @test padded[1:3] == [[0.2, 0.2], [0.3, 0.3], [0.1, 0.1]]
    @test length(padded) == 4 && all(0 .<= padded[4] .<= 1)
    @test length(BigHOPs._members(s, runs, 27, 1, 2)) == 1

    for _ in 1:20
        trial = BigHOPs._evolve(s, runs, 2, 27, 27, 1)
        @test length(trial) == 2
        @test all(0 .<= trial .<= 1)
    end
end

@testset "DEHB promotes the best unpromoted configurations from the level below" begin
    @info "Testing DEHB's promotion choice: ranked by value at the level below, skipping configurations already run at the higher level, falling back to the best"

    completed(id, r, u, v) = BigHOPs._with_result(BigHOPs.RunEntry(id, (r=r, a=0.0), u, Dict{Symbol,Any}()), BigHOPs.Completed, v, nothing)
    runs = [completed(i, 1, [0.1i], v) for (i, v) in enumerate([3.0, 1.0, 2.0, 4.0])]
    s = DEHB(R=9, η=3, r_min=1)
    de = BigHOPs._subpopulations!(s)
    de.subpops[1][1:4] = 1:4

    bracket = BigHOPs._make_bracket(s, 1, 1)
    decision = BigHOPs._promote(bracket, bracket.rungs[1], 0)

    @test BigHOPs._promotion_candidate(s, runs, decision) == 2
    @test BigHOPs._promotion_candidate(s, runs, decision) == 2
    push!(runs, completed(5, 3, copy(runs[2].unit_params), 0.5))
    @test BigHOPs._promotion_candidate(s, runs, decision) == 3
    push!(runs, BigHOPs.RunEntry(6, (r=3, a=0.0), copy(runs[3].unit_params)))
    @test BigHOPs._promotion_candidate(s, runs, decision) == 1

    append!(runs, [BigHOPs.RunEntry(6 + j, (r=3, a=0.0), copy(runs[i].unit_params)) for (j, i) in enumerate((1, 4))])
    @test BigHOPs._promotion_candidate(s, runs, decision) == 2
end

@testset "DEHB draws the first rung of the first bracket uniformly at random" begin
    @info "Testing that bracket 1 of iteration 1 evaluates plain uniform draws from DEHB's rng at its first rung, as in the paper"

    ho = Hyperoptimizer(p -> p.a + p.b, (a=Continuous(0, 1), b=Continuous(0, 1)), DEHB(R=9, η=3, r_min=1, rng=StableRNG(5)))
    run!(ho; show_progress=false)
    rng = StableRNG(5)
    first_rung = filter(e -> e.metadata[:iteration] == 1 && e.metadata[:bracket] == 1 && e.metadata[:rung] == 1, ho.runs)
    @test [e.unit_params for e in first_rung] == [rand(rng, 2) for _ in 1:9]
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
        @test [e.metadata[:slot] for e in ho.runs if e.params.r == r] == [mod(i - 1, length(de.subpops[r])) + 1 for i in 1:count(e -> e.params.r == r, ho.runs)]
        @test all(id -> 1 <= id <= ho.n, filter(!=(0), de.subpops[r]))
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
    for (r, ids) in de.subpops, (slot, id) in enumerate(ids)
        id == 0 && continue
        entry = ho.runs[id]
        @test entry.status == BigHOPs.Completed
        @test entry.params.r == r
        @test entry.metadata[:slot] == slot

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
    for (r, ids) in de.subpops, (slot, id) in enumerate(ids)
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
        @test [e.metadata[:slot] for e in ho.runs if e.params.r == r] == [mod(i - 1, length(de.subpops[r])) + 1 for i in 1:count(e -> e.params.r == r, ho.runs)]
    end
end
