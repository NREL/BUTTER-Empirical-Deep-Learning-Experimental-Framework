#!/bin/bash

#SBATCH --job-name=dmp
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=3
#SBATCH --tasks-per-node=1
#SBATCH --time=59
#SBATCH --account=dmpapps
#SBATCH --partition=debug
#SBATCH --cpu-freq=high-high:Performance

# Eagle Modules
source ./admp
unset LD_PRELOAD

num_nodes=$(($SLURM_JOB_NUM_NODES))
#total_cpus=$(($SLURM_JOB_NUM_NODES * $SLURM_CPUS_ON_NODE))
echo "num_nodes =" $num_nodes

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

echo "nodes: " $nodes
echo "nodes_array: " $nodes_array

for ((i = 0; i <= $num_nodes; i++)); do
  node=${nodes_array[$i]}
  echo "srun --nodes=1 --ntasks=1 -w $node $* &"
  srun --nodes=1 --ntasks=1 -w $node "$@" &
done

# sbatch -N1 -t59 python -m dmp.jobqueue.jq_node_manager dmp charles_test "[[0,2,0,1], [2,34,0,0], [34,36,1,2]]"
# python -m dmp.jobqueue.jq_node_manager dmp test_queue "[[0,2,0,1], [2,6,0,0]]"


#
#for (( RANK=1; RANK<=$DMP_NUM_GPU_WORKERS; RANK+=1 )); do
#    DMP_TYPE=GPU DMP_RANK=$RANK python -u -m dmp.jq_runner ${1} ${2} &
#done
#
#for (( RANK=1; RANK<=$DMP_NUM_CPU_WORKERS; RANK+=1 )); do
#    CUDA_VISIBLE_DEVICES=-1 DMP_TYPE=CPU DMP_RANK=$RANK python -u -m dmp.jq_runner ${1} ${2} &
#done
#
## Launch the dmp queue runner
##srun python -u -m dmp.jq_runner ${1} ${2}
#
##srun --time=59 --account=dmpapps --partition=debug --gres=gpu:1 --nodes=1 --pty $SHELL
