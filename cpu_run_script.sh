#!/bin/bash
ARGS=("$@")
node_list=("${ARGS[0]}")
core_list=("${ARGS[1]}")
command=("${ARGS[@]:2}")


echo args "${ARGS[@]}"
echo node_list "$node_list"
echo core_list "$core_list"

DMP_SCRIPT_DIR="$DMP_DIR/script/$DMP_CONFIG/"
source $DMP_SCRIPT_DIR/dmp_activate_cpu.bash

export GOMP_CPU_AFFINITY="$core_list"
export OMP_PROC_BIND=CLOSE
export OMP_SCHEDULE=STATIC
export OMP_DISPLAY_AFFINITY=TRUE

export TF_ENABLE_MKL_NATIVE_FORMAT=1
export TF_ENABLE_ONEDNN_OPTS=1
export KMP_AFFINITY=verbose,warnings,granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=TRUE
export TF_ENABLE_XLA=1

# export ZENDNN_MEMPOOL_ENABLE=1
# export ZENDNN_TENSOR_POOL_LIMIT=16
# export ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=0
# export ZENDNN_BLOCKED_FORMAT=0
# export ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024

echo executing numactl --cpunodebind="$node_list" --membind="$node_list" --physcpubind="$core_list" "${command[@]}"
numactl --cpunodebind="$node_list" --membind="$node_list" --physcpubind="$core_list" "${command[@]}"
