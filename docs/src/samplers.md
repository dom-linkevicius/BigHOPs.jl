# Samplers

A sampler decides which hyperparameters each trial gets. BigHOPs.jl has two kinds:

- **Full-budget samplers** evaluate every trial once, at full cost. They are given the number of trials up front: `Hyperoptimizer(objective, candidates; sampler=..., n=...)`.
- **Successive-halving samplers** start many trials on a small resource (epochs, samples, iterations) and give more only to the best ones. The number of trials comes from their schedule: `Hyperoptimizer(objective, candidates, sampler)`.

Every sampler is seeded by default, so repeating a search proposes the same trials.

## Full-budget samplers

### RandomSampler

```julia
RandomSampler(rng=StableRNG(1))
```

Draws each trial's hyperparameters independently at random. It is the simplest baseline, and also the default inner sampler of `Hyperband` and `ASHA`.

### LHSampler

```julia
LHSampler(; rng=StableRNG(1))
```

Draws all `n` trials up front as a Latin hypercube, which spreads them evenly along every hyperparameter. It warns if `n` is too small to cover every combination of discrete values. Each `LHSampler()` can back only one `Hyperoptimizer`.

### SobolSampler

```julia
SobolSampler(; rng=StableRNG(1))
```

Takes each trial as the next point of a Sobol sequence, shifted by one random offset, so successive trials spread over the space more evenly than independent random draws. Unlike `LHSampler`, it needs no fixed number of trials; after any power-of-2 number of trials, each hyperparameter's range is covered evenly.

### GPSampler

```julia
GPSampler(gp; next_point, noise, n_initial, initial=SobolSampler(), fit_hyperparams=nothing,
          standardize=BigHOPs.zscore, strict_initialization=true, rng=StableRNG(1))
```

Models the objective as a Gaussian process over the hyperparameters. The first `n_initial` trials come from `initial`; after that, the GP is conditioned on every completed trial, and `next_point` chooses each new trial from the posterior. `ThompsonSampling(n_candidates; jitter)` is the provided `next_point`: it draws the posterior at `n_candidates` Sobol points and returns the one where the draw is lowest.

```julia
using AbstractGPs

sampler = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(512; jitter=1e-6),
                    noise=1e-6, n_initial=10)
ho = Hyperoptimizer(objective, candidates; sampler, n=50)
```

`fit_hyperparams(X, y)` can refit the kernel's hyperparameters, and optionally the noise, before each conditioning. See the [API](api.md) for the remaining options.

## Successive-halving samplers

These samplers need an objective with a resource knob: the objective receives the resource level as `p.r`, and a larger `p.r` should mean a more expensive but more accurate evaluation, such as more training epochs. They share one schedule:

- `R` is the largest resource level, given to the trials that survive every round;
- `r_min` is the smallest resource level, where the most aggressive bracket starts;
- `η` is the factor between rounds: after each round, the best `1/η` of the trials continue with `η` times the resource;
- `iterations` repeats the whole schedule.

A bracket starts some number of trials at one resource level and halves its way up to `R`, and the schedule runs several brackets, from aggressive (many trials, starting at `r_min`) to conservative (a few trials, all at `R`).

When a trial is promoted to a higher resource level, it does not have to start again from scratch. Wrap the objective in `Stateful`, and it receives the promoted trial's earlier state as `pre_artefact`, so training can continue where it stopped:

```julia
function objective(p; pre_artefact=nothing)
    model, epochs_done = pre_artefact === nothing ? (new_model(p), 0) : pre_artefact
    train!(model, p.r - epochs_done)
    return validation_loss(model), (model, p.r)
end

ho = Hyperoptimizer(Stateful(objective), candidates, Hyperband(R=81, η=3, r_min=1))
```

`Hyperband`, `ASHA` and `BOHB` pass the state on; `DEHB` trains every trial from scratch.

### Hyperband

```julia
Hyperband(; R, η=3, r_min=1, iterations=1, inner=RandomSampler())
```

Successive halving as in Li et al. (2018): a rung must finish before any of its trials are promoted. `inner` draws the starting configurations, at random by default. When every open bracket is waiting on results and the executor has spare capacity, the next bracket starts.

### ASHA

```julia
ASHA(; R, η=3, r_min=1, iterations=1, inner=RandomSampler())
```

Asynchronous `Hyperband` (Li et al., 2020), with the same schedule and arguments: a trial is promoted as soon as it is among the best `1/η` completed so far in its rung, instead of waiting for the rung to finish. This keeps workers busy under `Threaded` or `DistributedQueue` when trial durations vary.

### DEHB

```julia
DEHB(; R, η=3, r_min=1, iterations=1, F=0.5, crossover=0.5, rng=StableRNG(1))
```

`Hyperband`'s schedule with differential evolution in place of random draws (Awad et al., 2021). Each resource level keeps a subpopulation of configurations; new trials are mutated and crossed over from them, so later trials build on the ones that did well. `F` is the mutation factor and `crossover` the crossover probability. Every trial trains from scratch, so a `Stateful` objective never receives a `pre_artefact`.

### BOHB

```julia
BOHB(; R, η=3, r_min=1, iterations=1, ρ=1/3, q=0.15, N_s=64, N_min=nothing, b_w=3,
     min_bandwidth=1e-3, rng=StableRNG(1))
```

`Hyperband`'s schedule with the first rung of each bracket drawn from kernel density estimates (Falkner et al., 2018), while promotions are as in `Hyperband`. Once a resource level has enough completed trials, they are split into the best ones and the rest, a density is fitted to each, and the next configuration is the one among `N_s` candidates that is most likely under the good density relative to the bad one. A fraction `ρ` of draws stays uniformly random, so the search keeps exploring.

The model needs more completed trials as the number of hyperparameters `d` grows: by default `N_min = d + 1`, and both the good and the bad set need more than `d` trials, so a resource level gets a model only after about `2d + 2` trials there.
