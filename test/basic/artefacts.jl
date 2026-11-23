mutable struct FakeNetwork
    weight::Float64
end

function train_step(p; pre_artefact=nothing)
    net = pre_artefact === nothing ? FakeNetwork(0.0) : pre_artefact
    net.weight += p.lr * p.momentum
    loss = abs(net.weight - 1.0)
    return loss, net
end

struct LoggingWrapper{F}
    f::F
end
const CALL_OBJECTIVE_LOG = Any[]
BigHOPs.call_objective(w::LoggingWrapper, params, pre_artefact) = (push!(CALL_OBJECTIVE_LOG, pre_artefact); w.f(params))

@testset "Stateful objectives / artefacts" begin
    @info "Testing Stateful objective artefact threading"

    ho_plain = Hyperoptimizer(p -> p.a + p.b, (a=Nominal([1, 2]), b=Nominal([3, 4])); n=3)
    run!(ho_plain)
    @test all(e -> e.pre_artefact === nothing, ho_plain.runs)
    @test all(e -> e.post_artefact === nothing, ho_plain.runs)

    ho = Hyperoptimizer(Stateful(train_step), (lr=Nominal([0.1, 0.5, 1.0]), momentum=Nominal([1.0])); n=3)
    run!(ho)
    @test all(v -> v isa Float64, results(ho))
    @test all(e -> e.pre_artefact === nothing, ho.runs)
    @test all(e -> e.post_artefact isa FakeNetwork, ho.runs)
    @test minimum(ho) isa Float64
    @test ho.runs[ho.best_min_id].post_artefact isa FakeNetwork

    ho_nan = Hyperoptimizer(Stateful((p; pre_artefact=nothing) -> (NaN, "unused")), (a=Nominal([1]),); n=1)
    @test_logs (:warn, r"NaN") run!(ho_nan)
    @test length(results(ho_nan)) == 0

    ho_err = Hyperoptimizer(Stateful((p; pre_artefact=nothing) -> error("boom")), (a=Nominal([1]),); n=1)
    @test_logs (:warn, r"non-Real") run!(ho_err)
    @test length(results(ho_err)) == 0

    ho_custom = Hyperoptimizer(LoggingWrapper(p -> p.a + p.b), (a=Nominal([1]), b=Nominal([2])); n=1)
    run!(ho_custom)
    @test CALL_OBJECTIVE_LOG == [nothing]
    @test minimum(ho_custom) == 3

    ho_tuple = Hyperoptimizer(p -> (p.a + p.b, "diagnostic"), (a=Nominal([1]), b=Nominal([2])); n=1)
    @test_logs (:warn, r"non-Real") run!(ho_tuple)
    @test length(results(ho_tuple)) == 0
    @test ho_tuple.runs[1].status == BigHOPs.Failed
    @test ho_tuple.runs[1].post_artefact === nothing

    let n_calls = Ref(0)
        global missing_after_first(p) = (n_calls[] += 1; n_calls[] == 1 ? 5.0 : missing)
    end
    ho_missing = Hyperoptimizer(missing_after_first, (a=Nominal([1, 2]),); n=2)
    @test_logs (:warn, r"non-Real") run!(ho_missing)
    @test minimum(ho_missing) == 5.0
    @test length(results(ho_missing)) == 1
    @test ho_missing.runs[2].status == BigHOPs.Failed
end
