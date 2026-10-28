@testset "DataFrame" begin
    @info "Testing DataFrame(ho)"

    ho = Hyperoptimizer(p -> p.a^2, (a=Nominal([1, 2, 3]),); n=3)
    run!(ho; show_progress=false)
    df = DataFrame(ho)
    @test size(df) == (3, 8)
    @test names(df) == ["id", "status", "a", "value", "error", "pre_artefact", "post_artefact", "metadata"]
    @test df.id == [1, 2, 3]
    @test all(df.status .== BigHOPs.Completed)
    @test all(df.value .== df.a .^ 2)
    @test all(df.error .=== nothing)
    @test all(df.pre_artefact .=== nothing)
    @test all(df.post_artefact .=== nothing)
    @test all(isempty, df.metadata)

    let n_calls = Ref(0)
        global dataframe_nan_after_first(p) = (n_calls[] += 1; n_calls[] == 1 ? p.a : NaN)
    end
    ho_nan = Hyperoptimizer(dataframe_nan_after_first, (a=Nominal([1, 2]),); n=2)
    @test_logs (:warn, r"NaN") run!(ho_nan; show_progress=false)
    df_nan = DataFrame(ho_nan)
    @test size(df_nan) == (2, 8)
    @test count(==(BigHOPs.Failed), df_nan.status) == 1
    @test ismissing(df_nan.value[df_nan.status.==BigHOPs.Failed][1])
    @test isnan(df_nan.error[df_nan.status.==BigHOPs.Failed][1])

    ho_stateful = Hyperoptimizer(Stateful((p; pre_artefact=nothing) -> (p.a, "artefact-$(p.a)")), (a=Nominal([1, 2]),); n=2)
    run!(ho_stateful; show_progress=false)
    df_stateful = DataFrame(ho_stateful)
    @test df_stateful.post_artefact == ["artefact-$a" for a in df_stateful.a]

    ho_multi = Hyperoptimizer(p -> p.a + p.b, (a=Nominal([1]), b=Nominal([2])); n=1)
    run!(ho_multi; show_progress=false)
    @test names(DataFrame(ho_multi)) == ["id", "status", "a", "b", "value", "error", "pre_artefact", "post_artefact", "metadata"]

    ho_sh = Hyperoptimizer(Stateful((p; pre_artefact=nothing) -> (p.a, nothing)), (a=Nominal([1, 2]),), Hyperband(R=3, η=3, r_min=1))
    run!(ho_sh; show_progress=false)
    df_sh = DataFrame(ho_sh)
    @test df_sh.metadata == [e.metadata for e in ho_sh.runs]
    @test all(m -> haskey(m, :rung), df_sh.metadata)
    df_sh.metadata[1][:rung] = -1
    @test ho_sh.runs[1].metadata[:rung] != -1

    ho_empty = Hyperoptimizer(nothing, (a=Nominal([1]),); n=1)
    @test size(DataFrame(ho_empty)) == (0, 0)
end
