#!/bin/bash
#SBATCH --job-name=bigho_nn_threaded
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%A_%a.log

julia --project=. --threads="$SLURM_CPUS_PER_TASK" run_task.jl nn "$SLURM_ARRAY_TASK_ID"
