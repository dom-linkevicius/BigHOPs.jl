# GPSampler: fitting the hyperparameters

Examples of `fit_hyperparams` for `GPSampler`, kept here for the eventual documentation.

Both examples maximize the likelihood on every tell using Optim and Zygote, which BigHOPs does not depend on. They optimize logarithms within bounds, which keeps every value positive and stops the fit collapsing onto degenerate solutions while there are few trials.

```julia
# Refits the kernel hyperparameters only; fit_kernel returns a GP, so the noise stays fixed at `noise`.
using BigHOPs, AbstractGPs, Optim, Zygote

noise = 1e-4
kernel(lengthscales, variance) = variance * with_lengthscale(Matern52Kernel(), lengthscales)
negative_log_likelihood(X, y, lengthscales, variance, noise) = -logpdf(GP(kernel(lengthscales, variance))(X, noise), y)

function minimize_within(f, lower, upper)
    result = optimize(f, v -> only(Zygote.gradient(f, v)), lower, upper, (lower .+ upper) ./ 2, Fminbox(LBFGS()); inplace=false)
    return Optim.minimizer(result)
end

function fit_kernel(X, y)
    d = length(first(X))
    v = minimize_within(v -> negative_log_likelihood(X, y, exp.(v[1:d]), exp(v[d+1]), noise),
                        fill(log(1e-2), d + 1), fill(log(1e2), d + 1))
    return GP(kernel(exp.(v[1:d]), exp(v[d+1])))
end

sampler = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(10; jitter=1e-6), noise=noise, n_initial=10,
                    fit_hyperparams=fit_kernel)
```

```julia
# Refits the kernel hyperparameters and the noise variance; fit_kernel_and_noise returns a (gp, noise) pair, so `noise` can be nothing.
using BigHOPs, AbstractGPs, Optim, Zygote

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

sampler = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(10; jitter=1e-6), noise=nothing, n_initial=10,
                    fit_hyperparams=fit_kernel_and_noise)
```
