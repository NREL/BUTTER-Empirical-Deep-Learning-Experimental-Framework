#!/bin/bash
ARGS=("$@")
num_sockets=("${ARGS[0]}")
num_cores=("${ARGS[1]}")
socket_list=("${ARGS[2]}")
core_list=("${ARGS[3]}")
command=("${ARGS[@]:4}")

export OMP_NUM_THREADS=$()

echo "${ARGS[@]}"
echo "$num_sockets"
echo "$num_cores"
echo "$socket_list"
echo "$core_list"
echo "${command[@]}"

numactl --cpunodebind="$socket_list" --preferred="$socket_list" --physcpubind="$core_list" "${command[@]}"
