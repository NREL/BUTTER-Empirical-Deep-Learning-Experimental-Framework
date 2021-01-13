#!/bin/bash

#SBATCH --job-name=dmp
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=120
#SBATCH --account=dmpapps
#SBATCH --cpu-freq=high-high:Performance

# MODIFY HERE according to your environment setup
source ~/admp
unset LD_PRELOAD

echo "executing command... python -u -m "$@"

python -u -m "$@"

