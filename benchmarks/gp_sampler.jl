using AbstractGPs
using Optim: Optim, LBFGS, optimize
import Zygote

kernel(lengthscales, variance) = variance * with_lengthscale(Matern52Kernel(), lengthscales)
negative_log_likelihood(X, y, lengthscales, variance, noise) = -logpdf(GP(kernel(lengthscales, variance))(X, noise), y)

to_box(v, lower, upper) = lower .* (upper ./ lower) .^ (1 ./ (1 .+ exp.(-v)))

function fit_kernel_and_noise(X, y)
    d = length(first(X))
    lower = [fill(1e-2, d + 1); 1e-6]
    upper = [fill(1e2, d + 1); 1.0]
    function nll(v)
        p = to_box(v, lower, upper)
        return negative_log_likelihood(X, y, p[1:d], p[d+1], p[d+2])
    end
    v = Optim.minimizer(optimize(nll, v -> only(Zygote.gradient(nll, v)), zeros(d + 2), LBFGS(); inplace=false))
    p = to_box(v, lower, upper)
    return GP(kernel(p[1:d], p[d+1])), p[d+2]
end

gp_sampler(rng) = GPSampler(
    GP(Matern52Kernel()); 
    next_point=ThompsonSampling(1000; jitter=1e-6), 
    noise=nothing, 
    n_initial=10,
    fit_hyperparams=fit_kernel_and_noise, 
    rng=rng
)
