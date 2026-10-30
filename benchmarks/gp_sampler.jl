using AbstractGPs
using Optim: Optim, LBFGS, optimize
using ParameterHandling: positive, value_flatten
using StableRNGs: StableRNG
import Zygote

const GP_NOISE = 1e-4
const GP_N_INITIAL = 10
const GP_NEXT_POINT = ThompsonSampling(1000; jitter=1e-6)

gp_kernel(θ) = θ.variance * with_lengthscale(Matern52Kernel(), θ.lengthscales)

function fit_gp(X, y)
    flat, unflatten = value_flatten((lengthscales=positive(ones(size(X.X, 1))), variance=positive(1.0)))
    nlml(v) = -logpdf(GP(gp_kernel(unflatten(v)))(X, GP_NOISE), y)
    result = optimize(nlml, v -> only(Zygote.gradient(nlml, v)), flat, LBFGS(); inplace=false)
    return GP(gp_kernel(unflatten(Optim.minimizer(result))))
end

gp_sampler(rng=StableRNG(1)) =
    GPSampler(GP(Matern52Kernel()); next_point=GP_NEXT_POINT, noise=GP_NOISE, n_initial=GP_N_INITIAL, fit=fit_gp, rng=rng)
