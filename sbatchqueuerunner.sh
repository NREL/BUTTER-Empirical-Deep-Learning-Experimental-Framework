#!/bin/bash

#SBATCH --job-name=dmp
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#--SBATCH --tasks-per-node=8
#SBATCH --time=59
#SBATCH --account=dmpapps
#SBATCH --partition=debug
#SBATCH --cpu-freq=high-high:Performance

# Eagle Modules
module purge
module use /nopt/nrel/apps/modules/centos74/modulefiles/
module load gcc/7.4.0
module load cuda/10.0.130
module load cudnn/7.4.2/cuda-10.0
module load conda

# Conda
source ~/.bashrc
conda activate dmp
unset LD_PRELOAD

# Extra Stuff
#export TEST_TMPDIR=/scratch/ctripp/tmp
#export TMPDIR=/scratch/ctripp/tmp
#export PYTHONPATH=/scratch/ctripp/src/dmp
#cd /projects/dmpapps/ctripp/src

# Launch the dmp queue runner
srun python -u -m dmp.jq_runner ${1} ${2}
