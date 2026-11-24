#!/bin/bash
#SBATCH --job-name=bighops_counting_ones
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=results/counting_ones/logs/%x_%A_%a.log

julia --project=. --threads="$SLURM_CPUS_PER_TASK" run_task.jl counting_ones "$SLURM_ARRAY_TASK_ID"
