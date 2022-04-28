import argparse
import math
import platform
import select
import sys
import dmp.jobqueue_interface.node_manager as node_manager
import subprocess
import os


def get_run_script(default_script, custom_script):
    if os.path.exists(custom_script):
        return custom_script
    return default_script


def make_worker_process(rank, command):
    print(f'Creating subprocess {rank} with command: "{" ".join(command)}"')
    return subprocess.Popen(
        command, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        close_fds=True)


def run_worker(run_script, project, queue, cores_per_cpu, workers, config):
    start_core = config[2]
    end_core = start_core + config[3]
    physcpus = ','.join([str(i) for i in range(start_core, end_core+1)])
    start_socket = int(start_core / cores_per_cpu)
    end_socket = int(end_core / cores_per_cpu)
    socket_list = ','.join([str(i)
                            for i in range(start_socket, end_socket+1)])
    num_sockets = end_socket - start_socket + 1

    command = [
        run_script,
        'numactl', f'--cpunodebind={socket_list}', f'--preferred={socket_list}',
        f'--physcpubind={physcpus}',
        'python', '-u', '-m', 'dmp.jobqueue_interface.worker_manager',
        'python', '-u', '-m', 'dmp.jobqueue_interface.worker',
        start_socket, num_sockets, *config, args.project, args.queue]
    return make_worker_process(len(workers), command)


def main():
    args = sys.argv
    project = args[1]
    queue = args[2]

    host = platform.node()

    print(
        f'Started Node Manager on host "{host}" for project "{project}" and queue "{queue}".')
    print(f'Launching worker processes...')

    total_cpu_cores = int(subprocess.check_output(
        'grep -c processor /proc/cpuinfo', shell=True))
    cpu_sockets = int(subprocess.check_output(
        'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
    cores_per_cpu = int(total_cpu_cores / cpu_sockets)

    gpu_mems = [int(i) for i in subprocess.check_output(
        'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader', shell=True).splitlines()]

    min_gpu_mem_per_worker = 4096
    worker_gpu_mem_overhead = 1024
    min_total_worker_gpu_mem = min_gpu_mem_per_worker + worker_gpu_mem_overhead

    min_gpu_mem_buffer = 500
    max_worker_per_gpu = 4

    cores_per_gpu_worker = 1
    min_cores_per_cpu_worker = 1

    cores_remaining_per_socket = [cores_per_cpu for _ in range(cpu_sockets)]
    gpu_worker_configs = []
    for gpu_number, gpu_mem in enumerate(gpu_mems):
        mem_avail = gpu_mem - min_gpu_mem_buffer
        if mem_avail < 0:
            continue

        num_workers = max(max_worker_per_gpu,
                          int(math.floor(mem_avail / min_total_worker_gpu_mem)))

        mem_per_worker = (mem_avail / num_workers) - worker_gpu_mem_overhead

        for _ in range(num_workers):
            worker_count = len(gpu_worker_configs)
            socket = worker_count % cpu_sockets
            core = (socket * cores_per_cpu + int((worker_count /
                    cpu_sockets) * cores_per_gpu_worker)) % cores_per_cpu
            cores_remaining_per_socket[socket] -= cores_per_gpu_worker
            gpu_worker_configs.append(
                [core, cores_per_gpu_worker, gpu_number, 1, mem_per_worker])

    cpu_worker_configs = []
    for socket, cores_remaining in enumerate(cores_remaining_per_socket):
        if cores_remaining < min_cores_per_cpu_worker:
            continue
        core = cores_per_cpu - cores_remaining
        cpu_worker_configs.append([core, cores_remaining, 0, 0, 0])

    gpu_run_script = get_run_script(
        'gpu_run_script.sh', 'custom_gpu_run_script.sh')
    cpu_run_script = get_run_script(
        'cpu_run_script.sh', 'custom_cpu_run_script.sh')

    workers = []
    for config in gpu_worker_configs:
        workers.append(run_worker(gpu_run_script, project, queue,
                       cores_per_cpu, workers, config))

    for config in cpu_worker_configs:
        workers.append(run_worker(cpu_run_script, project, queue,
                       cores_per_cpu, workers, config))

    streams = [w.stdout for w in workers]
    stream_name_map = {id(s): f'{i}:' for i, s in enumerate(streams)}

    def output(stream, line):
        if len(line) == 0:
            return
        name = stream_name_map[id(stream)]
        if not isinstance(line, str):
            line = line.decode("utf-8")
        line = name + line
        sys.stdout.write(line)
        sys.stdout.flush()

    print('Starting output redirection...')
    while True:
        rstreams, _, _ = select.select(streams, [], [], 30)
        exit = False
        for stream in rstreams:
            line = stream.readline()
            if len(line) == 0:
                exit = True
            output(stream, line)
        if (len(rstreams) == 0 or exit) and all(w.poll() is not None for w in workers):
            break

    for stream in streams:
        while True:
            line = stream.readline()
            if len(line) == 0:
                break
            output(stream, line)

    print(f'Waiting for worker processes to exit...')
    for worker in workers:
        worker.wait()
    print('Exiting Worker Manager...')


if __name__ == "__main__":
    main()
