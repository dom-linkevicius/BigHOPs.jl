# Benchmarks

The benchmarks run as Slurm array jobs on a cluster; the results are then downloaded and plotted locally. Every command below is run from this `benchmarks/` folder.

```
benchmarks/
  tasks.jl        every repeat of every benchmark, with its result file and sbatch file
  submit.jl       submits the repeats as Slurm arrays
  run_task.jl     runs one array job, called by the sbatch files
  benchmark_code/
    counting_ones/  counting_ones.jl, counting_ones.sbatch, plot_counting_ones.jl
    nn/             nn.jl, nn_threaded.sbatch, nn_serial.sbatch, plot_results.jl
  results/        one folder per benchmark: results and Slurm logs (gitignored)
```

## Running on the cluster

Get the repository onto the cluster with git, so the sbatch files keep Unix line endings, and make sure `julia` is on the `PATH` of the compute nodes. Then, on a login node:

```sh
julia --project=. submit.jl                 # repeats without a result yet
julia --project=. submit.jl counting_ones   # delete this benchmark's results, rerun all of it
julia --project=. submit.jl all             # delete all results, rerun everything
```

`submit.jl` installs and precompiles the environment before submitting. The jobs inherit the environment of the shell they are submitted from, so set these variables before calling it:

- `BIGHOPS_BENCHMARK_DEPLOY=true`: run the NN benchmark at full scale; without it, it runs a small test version.
- `BIGHOPS_BENCHMARK_REPEATS`: NN repeats, 50 by default at full scale.
- `BIGHOPS_COUNTING_ONES_REPEATS`: counting ones repeats, 512 by default.

Each benchmark's Slurm settings (CPUs, time limit, memory) are in its sbatch files. A counting ones job runs 4 repeats, an NN job one. Each repeat is saved to `results/<benchmark>/<setting>/repeat_<k>.jld2` when it finishes, and the job's log to `results/<benchmark>/`.

A single job can also be run by hand, e.g. the first counting ones job:

```sh
julia --project=. run_task.jl counting_ones 1
```

## Plotting

Copy `results/` from the cluster into your local checkout, then:

```sh
julia --project=. benchmark_code/counting_ones/plot_counting_ones.jl
julia --project=. benchmark_code/nn/plot_results.jl
```

The figures are written to `docs/src/figures/`.
