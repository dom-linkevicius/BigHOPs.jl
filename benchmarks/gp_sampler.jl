using AbstractGPs
using LinearAlgebra
using NLSolversBase: only_fg!
using Optim: Optim, LBFGS, optimize

struct KernelAndNoiseFit
    warm::Bool
    every::Int
    start::Vector{Float64}
    fitted_at::Base.RefValue{Int}
end

KernelAndNoiseFit(; warm::Bool, every::Int=1) = KernelAndNoiseFit(warm, every, Float64[], Ref(0))

function negative_log_posterior!(G, U, y, w, μ, σ)
    d, n = size(U)
    lengthscales, noise = exp.(w[1:d]), exp(w[d+1])
    Z = U ./ lengthscales
    squared_norms = vec(sum(abs2, Z; dims=1))
    r = sqrt.(max.(squared_norms .+ squared_norms' .- 2 .* (Z' * Z), 0.0))
    decay = exp.(-sqrt(5) .* r)
    C = cholesky(Symmetric((1 .+ sqrt(5) .* r .+ (5 / 3) .* r .^ 2) .* decay + noise * I))
    α = C \ y
    if G !== nothing
        W = inv(C) - α * α'
        A = W .* (5 / 3) .* (1 .+ sqrt(5) .* r) .* decay
        d_lengthscales = (U .^ 2 * vec(sum(A; dims=2)) .- vec(sum((U * A) .* U; dims=2))) ./ lengthscales .^ 2
        G .= [d_lengthscales; noise * tr(W) / 2] .+ 1 .+ (w .- μ) ./ σ .^ 2
    end
    return dot(y, α) / 2 + logdet(C) / 2 + n * log(2π) / 2 + sum(w .+ (w .- μ) .^ 2 ./ (2 .* σ .^ 2))
end

function (fit::KernelAndNoiseFit)(X, y)
    U = X.X
    d = size(U, 1)
    if isempty(fit.start) || length(y) - fit.fitted_at[] >= fit.every
        μ = [fill(sqrt(2) + log(d) / 2, d); -4.0]
        σ = [fill(sqrt(3), d); 1.0]
        start = fit.warm && !isempty(fit.start) ? fit.start : μ .- σ .^ 2
        fg!(F, G, w) = negative_log_posterior!(G, U, y, w, μ, σ)
        copy!(fit.start, Optim.minimizer(optimize(only_fg!(fg!), start, LBFGS())))
        fit.fitted_at[] = length(y)
    end
    return GP(with_lengthscale(Matern52Kernel(), exp.(fit.start[1:d]))), exp(fit.start[d+1])
end

gp_sampler(rng; warm=true, every=25) = GPSampler(
    GP(Matern52Kernel());
    next_point=ThompsonSampling(1000; jitter=1e-6),
    noise=nothing,
    n_initial=10,
    fit_hyperparams=KernelAndNoiseFit(; warm, every),
    rng=rng
)
