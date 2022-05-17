#!/bin/bash
ARGS=("$@")
num_sockets=("${ARGS[0]}")
num_cores=("${ARGS[1]}")
socket_list=("${ARGS[2]}")
core_list=("${ARGS[3]}")
command=("${ARGS[@]:4}")


echo args "${ARGS[@]}"
echo num_sockets "$num_sockets"
echo num_cores "$num_cores"
echo socket_list "$socket_list"
echo core_list "$core_list"


#export OMP_DYNAMIC=false
#export OMP_WAIT_POLICY=active
#export OMP_PROC_BIND=true
#export OMP_PLACES=cores
export OMP_NUM_THREADS=$((num_cores))
export OMP_DISPLAY_AFFINITY=TRUE
export KMP_AFFINITY=verbose
#export CRAY_OMP_CHECK_AFFINITY=TRUE

export TF_ENABLE_MKL_NATIVE_FORMAT=1
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

echo executing numactl --cpunodebind="$socket_list" --membind="$socket_list" --physcpubind="$core_list" "${command[@]}"
numactl --cpunodebind="$socket_list" --membind="$socket_list" --physcpubind="$core_list" "${command[@]}"
