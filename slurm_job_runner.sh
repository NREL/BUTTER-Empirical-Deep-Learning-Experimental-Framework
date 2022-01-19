#!/bin/bash

#SBATCH --job-name=dmp_test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=59
#SBATCH --account=dmpscale
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
