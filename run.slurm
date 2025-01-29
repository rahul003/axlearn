#!/bin/bash
#SBATCH --output=logs/%x/%j/out
#SBATCH --cpus-per-task 127

srun -l setup_node.sh
srun -l runner.sh