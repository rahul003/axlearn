#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --exclusive
#SBATCH --nodes=1


#Command : run_trn.sh <launch_script> <launch script options> 
#NOTE :  <launch script options> are defined by <launch_script>
#Example :
# run_gpu.sh runner_gpu.sh conda_env test_name setup.sh
srun  --kill-on-bad-exit=1 "$@"

