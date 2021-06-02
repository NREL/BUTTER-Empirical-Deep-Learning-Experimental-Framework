#!/bin/bash

#SBATCH --job-name=dmp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#--SBATCH --tasks-per-node=8
#SBATCH --time=59
#SBATCH --account=dmpapps
#SBATCH --partition=debug
#SBATCH --cpu-freq=high-high:Performance

# Eagle Modules
source ./admp
unset LD_PRELOAD

# Extra Stuff
#export TEST_TMPDIR=/scratch/ctripp/tmp
#export TMPDIR=/scratch/ctripp/tmp
#export PYTHONPATH=/scratch/ctripp/src/dmp
#cd /projects/dmpapps/ctripp/src

export DMP_NUM_GPU_WORKERS=1
export DMP_NUM_CPU_WORKERS=30

for (( RANK=1; RANK<=$DMP_NUM_GPU_WORKERS; RANK+=1 )); do
    DMP_TYPE=GPU DMP_RANK=$RANK python -u -m dmp.jq_runner ${1} ${2} &
done

for (( RANK=1; RANK<=$DMP_NUM_CPU_WORKERS; RANK+=1 )); do
    CUDA_VISIBLE_DEVICES=-1 DMP_TYPE=CPU DMP_RANK=$RANK python -u -m dmp.jq_runner ${1} ${2} &
done

# Launch the dmp queue runner
#srun python -u -m dmp.jq_runner ${1} ${2}

#srun --time=59 --account=dmpapps --partition=debug --gres=gpu:1 --nodes=1 --pty $SHELL