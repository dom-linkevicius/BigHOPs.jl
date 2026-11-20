# Domains

The high-level problem being solved in `BigHOPs.jl` is hyperparameter optimization. Letting $x$ be the hyperparameters of a model and $f(x)$ be the objective value with these parameters, we want to solve a minimization problem $\min_x f(x)$. Therefore, the initial step is to define the search space for the candidate values of $x$ (for now we omit how the candidates are sampled and how the objective is executed).

In `BigHOPs.jl` this candidate space is defined using a `NamedTuple`, with one entry per hyperparameter, and passed to the `Hyperoptimizer`. Using neural network training as a loose inspiration, an example of `domains` could be:
```julia
domains = (
    lr=Continuous(-4, -1; transform=exp10), # learning rate
    n_layers=Ordinal(1:3),                  # depth of network
    width=Ordinal([8, 16, 32, 64]),         # width of network
    activation=Nominal([tanh, abs]),        # activation functions
)
ho = Hyperoptimizer(f, domains, ...)
```

The objective `f` is evaluated by passing the sampled candidate values as a `NamedTuple` with instances from the domains specified in `domains`, e.g. `x.lr`, `x.width` and so on. Under the hood each domain (including `Ordinal` and `Nominal`) is sampled from a unit interval $[0, 1]$ with the domains specifying the transformations which take the sampled values to their natural domain. This allows for a unified candidate representation, overriding it only when a sampler requires it (see [BOHB](@ref)). `BigHOPs.jl` has three types of domains.

## Continuous

```@docs; canonical=false
Continuous
```

The `transform` makes it easy to specify hyperparameters that span orders of magnitude, such as learning rates or regularization strengths, which are best searched on a log scale.

## Ordinal

```@docs; canonical=false
Ordinal
```

## Nominal

```@docs; canonical=false
Nominal
```

The `Nominal` domain should be used when the choices have no intrinsic order, such as activation functions, optimizers or model architectures. The values can be anything, so the objective can use them directly, e.g. `x.activation.(y)` where `y` is some vector. In most cases `Nominal` behaves identically to `Ordinal`, except for the `BOHB` sampler, where a different kernel has to be used (see [BOHB](@ref)).