#!/bin/bash

#SBATCH --job-name=dmp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --tasks-per-node=1
#SBATCH --time=59
#SBATCH --account=dmpapps
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

for ((i = 1; i < $num_nodes; i++)); do
  node=${nodes_array[$i]}
  echo "srun --nodes=1 --ntasks=1 -w $node "$@" &"
  srun --nodes=1 --ntasks=1 -w $node "$@" &
done

echo ""$@""
"$@"

# [[0, 3, 0, 1, 3840], [3, 6, 0, 1, 3840], [6, 9, 0, 1, 3840], [9, 12, 0, 1, 3840], [12, 15, 1, 2, 3840], [15, 18, 1, 2, 3840], [18, 21, 1, 2, 3840], [21, 24, 1, 2, 3840], [24, 36, 0, 0, 0]]
# sbatch -N1 -t59 python -m dmp.jq.jq_node_manager dmp charles_test "[[0,2,0,1], [2,34,0,0], [34,36,1,2]]"
# sbatch -N1 -t59 slurm_job_runner.sh python -m dmp.jq.jq_node_manager dmp charles_test "[[0, 2, 0, 1], [1, 3, 0, 0], [2, 4, 0, 0], [3, 5, 0, 0], [4, 6, 0, 0], [5, 7, 0, 0], [6, 8, 0, 0], [7, 9, 0, 0], [8, 10, 0, 0], [9, 11, 0, 0], [10, 12, 0, 0], [11, 13, 0, 0], [12, 14, 0, 0], [13, 15, 0, 0], [14, 16, 0, 0], [15, 17, 0, 0], [16, 18, 0, 0], [17, 19, 0, 0], [18, 20, 0, 0], [19, 21, 0, 0], [20, 22, 0, 0], [21, 23, 0, 0], [22, 24, 0, 0], [23, 25, 0, 0], [24, 26, 0, 0], [25, 27, 0, 0], [26, 28, 0, 0], [27, 29, 0, 0], [28, 30, 0, 0], [29, 31, 0, 0], [30, 32, 0, 0], [31, 33, 0, 0], [32, 34, 0, 0], [33, 35, 1, 2]]"
# python -m dmp.jq.jq_node_manager dmp jordan_eagle_2 "[[0, 2, 0, 1]]"
# python -m dmp.jq.jq_node_manager dmp jordan_eagle_2 "[[0,2,0,1], [2,6,0,0]]"


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
