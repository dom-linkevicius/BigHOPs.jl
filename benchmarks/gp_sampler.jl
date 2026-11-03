using AbstractGPs
using Optim: Optim, LBFGS, Fminbox, optimize
import Zygote

kernel(lengthscales, variance) = variance * with_lengthscale(Matern52Kernel(), lengthscales)
negative_log_likelihood(X, y, lengthscales, variance, noise) = -logpdf(GP(kernel(lengthscales, variance))(X, noise), y)

function minimize_within(f, lower, upper)
    result = optimize(f, v -> only(Zygote.gradient(f, v)), lower, upper, (lower .+ upper) ./ 2, Fminbox(LBFGS()); inplace=false)
    return Optim.minimizer(result)
end

function fit_kernel_and_noise(X, y)
    d = length(first(X))
    v = minimize_within(v -> negative_log_likelihood(X, y, exp.(v[1:d]), exp(v[d+1]), exp(v[d+2])),
                        [fill(log(1e-2), d + 1); log(1e-6)], [fill(log(1e2), d + 1); 0.0])
    return GP(kernel(exp.(v[1:d]), exp(v[d+1]))), exp(v[d+2])
end

gp_sampler(rng) = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(1000; jitter=1e-6), noise=nothing, n_initial=10,
                            fit_hyperparams=fit_kernel_and_noise, rng=rng)
