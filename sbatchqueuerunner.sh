#!/bin/bash

#SBATCH --job-name=dmp
#SBATCH --nodes=2
#SBATCH --tasks-per-node=9
#SBATCH --time=59
#SBATCH --account=dmpapps
#SBATCH --partition=debug
#SBATCH --cpu-freq=high-high:Performance

# MODIFY HERE according to your environment setup

module purge
module use /nopt/nrel/apps/modules/centos74/modulefiles/
module load gcc/7.4.0
module load cuda/10.0.130
module load cudnn/7.4.2/cuda-10.0
module load conda

source ~/.bashrc

#export TEST_TMPDIR=/scratch/ctripp/tmp
#export TMPDIR=/scratch/ctripp/tmp
#export PYTHONPATH=/scratch/ctripp/src/dmp
#cd /projects/dmpapps/ctripp/src

conda activate dmp

unset LD_PRELOAD

python -u -m dmp.jq_runner dmp ${1}

