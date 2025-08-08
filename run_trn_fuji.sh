#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --job-name=mengchiy_test
#SBATCH --exclusive
#SBATCH --nodes=8
#SBATCH --partition dev
echo "PWD -"
pwd

echo "Git log - On commit - "
git log | head -n 3

echo "Printing diff"
git diff
echo "Done Printing diff"

# srun  --kill-on-bad-exit=1  run_mainline.sh "$@"
srun  --kill-on-bad-exit=1  runner_fuji.sh "$@"

#sbatch run_trn.sh 1 trn