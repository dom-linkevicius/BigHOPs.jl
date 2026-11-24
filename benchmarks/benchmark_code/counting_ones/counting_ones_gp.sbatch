#!/bin/bash
#SBATCH --job-name=bighops_counting_ones_gp
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --output=results/counting_ones/logs/%x_%A_%a.log

export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
julia --project=. --threads="$SLURM_CPUS_PER_TASK" run_task.jl counting_ones "$SLURM_ARRAY_TASK_ID"
