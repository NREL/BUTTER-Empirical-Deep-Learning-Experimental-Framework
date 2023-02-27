#!/bin/bash
ARGS=("$@")
node_list=("${ARGS[0]}")
core_list=("${ARGS[1]}")
command=("${ARGS[@]:2}")


echo args "${ARGS[@]}"
echo node_list "$node_list"
echo core_list "$core_list"


#export OMP_DYNAMIC=false
#export OMP_WAIT_POLICY=active
#export OMP_PROC_BIND=true
#export OMP_PLACES=cores
export OMP_NUM_THREADS=${#core_list[@]}
export OMP_DISPLAY_AFFINITY=TRUE
export KMP_AFFINITY=verbose
#export CRAY_OMP_CHECK_AFFINITY=TRUE

export TF_ENABLE_MKL_NATIVE_FORMAT=0
export TF_ENABLE_ONEDNN_OPTS=1
export KMP_AFFINITY=verbose,warnings,granularity=fine,compact,0,1
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
