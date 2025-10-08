#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --job-name=ishani_fuji1b_test
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=dev
srun --kill-on-bad-exit=1 ./run_script.sh