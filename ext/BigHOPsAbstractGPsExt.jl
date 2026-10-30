module BigHOPsAbstractGPsExt

using BigHOPs
using BigHOPs: GPSampler, ThompsonSampling, RunEntry, Completed
using AbstractGPs: ColVecs, posterior, mean, std

function _standardize(y)
    spread = std(y)
    return spread > 0 ? (y .- mean(y)) ./ spread : y .- mean(y)
end

function BigHOPs.on_tell!(s::GPSampler, runs, entry::RunEntry)
    entry.status === Completed || return nothing
    done = filter(e -> e.status === Completed, runs)
    X = ColVecs(stack(e.unit_params for e in done))
    y = _standardize(Float64[e.value for e in done])
    gp = s.fit === nothing ? s.gp : s.fit(X, y)
    s.posterior[] = posterior(gp(X, s.noise), y)
    return nothing
end

function (ts::ThompsonSampling)(post, d::Int, rng)
    candidates = rand(rng, d, ts.n_candidates)
    draw = rand(rng, post(ColVecs(candidates), ts.jitter))
    return candidates[:, argmin(draw)]
end

end
