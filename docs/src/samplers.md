# Samplers

Equipped with the domains defined in [Domains](@ref), we now present the machinery of how the next candidate is selected in `BigHOPs.jl`. Selection of the next candidate $x_{n+1}$ is handled by a sampler $\mathcal{S}$, which, depending on the sampler, may depend on the previous $x_1, \dots, x_n$ candidates or not. 

One way the samplers in `BigHOPs.jl` can be divided is by whether they evaluate the objective `f` at the full resource or not:
- **Full-resource samplers** evaluate `f` once, at the full resource (e.g. all training epochs), with the number of evaluations specified upfront. 
- **Successive-halving samplers** start (potentially many) trials at a small resource and promote only some fraction of the best trials at that resource level to the next stage, where they are evaluated with more resource. 

Another way to divide the samplers is by whether the sampler is adaptive, i.e. whether previous candidates inform the candidates sampled in the future. Therefore, we group the samplers into a $2 \times 2$ matrix by these two features:

|                        | Non-adaptive                                 | Adaptive         |
|:-----------------------|:---------------------------------------------|:-----------------|
| **Full-resource**      | `RandomSampler`, `LHSampler`, `SobolSampler` | `GPSampler`      |
| **Successive-halving** | `Hyperband`, `ASHA`                          | `DEHB`, `BOHB`   |

For reproducibility (in so far as it is possible under things like schedulers and differing resource contention at different reruns) each sampler has a default seed, which can be changed.

## Full-resource samplers

The number of objective evaluations for each full-resource sampler is specified by setting `n` when creating a `Hyperoptimizer`.

### Full-resource non-adaptive samplers

#### RandomSampler

```@docs; canonical=false
RandomSampler
```

The simplest baseline, and also the default inner sampler of `Hyperband` and `ASHA`.

#### LHSampler

```@docs; canonical=false
LHSampler
```

Uses the `QuasiMonteCarlo.jl` implementation.

#### SobolSampler

```@docs; canonical=false
SobolSampler
```

Uses the `Sobol.jl` implementation.

### Full-resource adaptive samplers

#### GPSampler

```@docs; canonical=false
GPSampler
```

`ThompsonSampling(n_candidates; jitter)` [Kandasamy2018](@cite) is the provided `next_point`: it draws the posterior at `n_candidates` Sobol points and returns the one where the draw is lowest.

```julia
using AbstractGPs

sampler = GPSampler(GP(Matern52Kernel()); next_point=ThompsonSampling(512; jitter=1e-6),
                    noise=1e-6, n_initial=10)
ho = Hyperoptimizer(f, domains; sampler, n=50)
```

## Successive-halving samplers

Successive halving as a hyperparameter optimization strategy was first introduced in [Jamieson2016](@citet), as a single bracket of rungs. [Li2018](@citet) extended it to hedge your bets given some budget, setting up a series of brackets split into rungs, where each successive rung evaluates the objective `f` with a progressively larger resource. With the total budget being approximately equal between brackets, the first bracket is the most aggressive, evaluating the objective many times with just enough resource, such that the good candidates would be distinguishable from the bad ones. The best fraction of the candidates is then promoted to the next rung and trained with a larger resource. The second bracket starts off with a higher minimal resource, following the same procedure, until the last bracket, where a few objective evaluations are made at the full resource. 

The computation given to a single evaluation of `f` is formalized via the notion of resource $r$ -- training epochs, amount of data, etc., whatever the problem context dictates. Other parameters used to set up the schedule for the brackets and the promotions are:

- ``R`` -- the largest resource level, given to the trials that survive every round;
- ``r_{min}`` -- the smallest resource level, where the most aggressive bracket starts;
- ``\eta`` -- the factor between rounds, after each round, the best $1/\eta$ of the trials continue with $\eta$ times the resource;
- ``k`` -- repeat the whole schedule that many times.

The number of brackets, rungs and trials follows [Li2018](@citet) definitions. With $s_{max} = \lfloor \log_\eta(R / r_{min}) \rfloor$, there are $s_{max} + 1$ brackets, indexed by $s = s_{max}, \dots, 0$ from the most to the least aggressive. Bracket $s$ has $s + 1$ rungs, and its first rung starts

```math
n_s = \left\lceil \frac{s_{max} + 1}{s + 1} \, \eta^s \right\rceil
```

trials at resource $r_s = r_{min} \eta^{s_{max} - s}$. Rung $i = 1, \dots, s + 1$ then runs $\lfloor n_s \eta^{-(i-1)} \rfloor$ trials at resource $r_s \eta^{i-1}$, so every bracket ends at $r_{min} \eta^{s_{max}}$. This equals $R$ only when $R / r_{min}$ is a power of $\eta$; otherwise a warning says how far short it falls. With $k$ repetitions, every bracket runs $k$ times.

Taking a specific example where $R=9$, $r_{min}=1$, $\eta=3$, we get $s_{max} = 2$ and three brackets:

| Bracket | Rung 1              | Rung 2              | Rung 3             |
|:--------|:--------------------|:--------------------|:-------------------|
| $s = 2$ | 9 trials at $r = 1$ | 3 trials at $r = 3$ | 1 trial at $r = 9$ |
| $s = 1$ | 5 trials at $r = 3$ | 1 trial at $r = 9$  |                    |
| $s = 0$ | 3 trials at $r = 9$ |                     |                    |

If every trial starts from scratch, the brackets spend a budget of $27$, $24$ and $27$. This information is also printed whenever a `Hyperoptimizer` is displayed:

```julia-repl
julia> ho = Hyperoptimizer(f, (x=Continuous(0, 1),), Hyperband(; R=9, η=3, r_min=1))
Hyperoptimizer with
  x [0.0, 1.0]
  resource levels: 1, 3, 9
  trials per level: 9, 8, 5
  total trials: 22
  no completed trials yet
```

Note that when a trial is promoted, it does not have to start from scratch. Using the `Stateful` wrapper over your objective allows `Hyperband`, `ASHA` and `BOHB` to continue training from where the trial finished in the previous rung, potentially saving a significant part of the budget, see [Stateful objectives](@ref).

!!! danger "`r` is reserved"
    Successive-halving samplers pass the resource level to the objective as `x.r`, in the same `NamedTuple` as the hyperparameters, so `r` cannot be the name of a hyperparameter.

Finally, contrary to the full-resource samplers, the number of objective evaluations of `f` is set by sampler's parameters, rather than `n`.

### Successive-halving non-adaptive samplers

#### Hyperband

```@docs; canonical=false
Hyperband
```

#### ASHA

```@docs; canonical=false
ASHA
```

### Successive-halving adaptive samplers

#### DEHB

```@docs; canonical=false
DEHB
```

#### BOHB

```@docs; canonical=false
BOHB
```

The model needs more completed trials as the number of hyperparameters $d$ grows. By default the KDE machinery starts at about $2d + 2$ completed trials at a specific resource level. Both the good and the bad set need more than $d$ trials since a KDE fitted on fewer points in $d$ dimensions is not very informative.
