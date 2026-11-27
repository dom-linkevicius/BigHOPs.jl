#!/bin/bash
#SBATCH --job-name=bighops_precompile
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=results/logs/%x_%j.log

export JULIA_NUM_PRECOMPILE_TASKS="$SLURM_CPUS_PER_TASK"
julia --project=. -e 'using Pkg; Pkg.precompile()'
