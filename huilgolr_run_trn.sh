#!/bin/bash
#SBATCH --output=logs/%x/%j/out
#SBATCH --cpus-per-task 127

export SCRIPT="huilgolr_run_trainer.sh"

source ../launchers/mnode_runner.sh