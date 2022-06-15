import math
import platform
import re
import select
import sys
import subprocess
import os


def get_run_script(default_script, custom_script):
    if os.path.exists(custom_script):
        return custom_script
    return default_script


def make_worker_process(rank, command):
    command = [str(a) for a in command]
    print(f'Creating subprocess {rank} with command: "{" ".join(command)}"')
    return subprocess.Popen(
        command, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        close_fds=True)


def run_worker(run_script, project, queue, workers, config):
    nodes = config[0]
    cores = config[1]
    num_nodes = len(nodes)
    num_cores = len(cores)
    core_list = ','.join([str(i) for i in cores])
    node_list = ','.join([str(i) for i in nodes])

    if queue == 0:
        command = [
            f'./{run_script}',
            num_nodes, num_cores, node_list, core_list,
            'echo',
            'python', '-u', '-m', 'dmp.jobqueue_interface.worker_manager',
            'python', '-u', '-m', 'dmp.jobqueue_interface.worker',
            nodes[0], num_nodes, cores[0], num_cores, config[2], config[3], config[4], project, queue]
    else:
        command = [
            f'./{run_script}',
            num_nodes, num_cores, node_list, core_list,
            'python', '-u', '-m', 'dmp.jobqueue_interface.worker_manager',
            'python', '-u', '-m', 'dmp.jobqueue_interface.worker',
            nodes[0], num_nodes, cores[0], num_cores, config[2], config[3], config[4], project, queue]

    return make_worker_process(len(workers), command)


def main():
    args = sys.argv
    project = args[1]
    queue = args[2]

    host = platform.node()

    print(
        f'Started Node Manager on host "{host}" for project "{project}" and queue "{queue}".')
    print(f'Launching worker processes...')

    # num_cores = int(subprocess.check_output(
    #     'grep -c processor /proc/cpuinfo', shell=True))
    # num_sockets = int(subprocess.check_output(
    #     'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))

    numa_hardware_output = subprocess.check_output(
        'numactl --hardware | grep -P "node \d+ cpus:"', shell=True).decode('ascii')

    numa_nodes = [s for s in numa_hardware_output.split(
        '\n') if s.startswith('node ')]

    numa_physcpubind_output = subprocess.check_output(
        'numactl --show | grep -P "physcpubind"', shell=True).decode('ascii')
    numa_cpus = {int(i) for i in [i.replace('\n', '').strip()
                                  for i in numa_physcpubind_output[len('physcpubind: '):].split(' ')]
                 if len(i) > 0}

    # numa_node_numbers = [int(re.search(r'\d+', numa_nodes[0]).group()) for s in numa_nodes]

    def group_hyperthreads(cpus):
        cpus = sorted(cpus)
        cpu_groups = []
        cpu_group = None
        for i, c in cpus:
            if i == 0 or c - cpus[i-1] > 1:
                cpu_group = []
                cpu_groups.append(cpu_group)
            cpu_group.append(c)
        return [v for p in zip(*cpu_groups) for v in p]


    numa_cores = [[i for i in
                   group_hyperthreads([int(i) for i in [i.replace('\n', '').strip()
                                     for i in n.split('cpus: ')[1].split(' ')] if len(i) > 0])
                   if i in numa_cpus]
                  for n in numa_nodes]

    print(f'numactl --hardware output: "{numa_hardware_output}".\n' +
          f'numactl --show output: "{numa_hardware_output}"\n' +
          f'NUMA topology: {numa_cores}',
          flush=True)

    # cores_per_node = len(numa_cores[0])

    gpu_mems = []
    try:
        gpu_mems = [int(i) for i in subprocess.check_output(
            'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader', shell=True).splitlines()]
    except subprocess.CalledProcessError:
        pass
    # num_cores = 32
    # num_sockets = 2
    # cores_per_socket = int(num_cores / num_sockets)
    # gpu_mems = [16*1024,]

    min_gpu_mem_per_worker = 6.5 * 1024
    worker_gpu_mem_overhead = 1024
    min_total_worker_gpu_mem = min_gpu_mem_per_worker + worker_gpu_mem_overhead

    min_gpu_mem_buffer = 500
    max_worker_per_gpu = 4

    cores_per_gpu_worker = 2
    min_cores_per_cpu_worker = 4
    target_cores_per_cpu_worker = 64

    # cores_allocated_per_node = [0 for _ in range(num_nodes)]
    cores_avaliable = numa_cores

    node = len(cores_avaliable) - 1
    gpu_worker_configs = []
    for gpu_number, gpu_mem in enumerate(gpu_mems):
        mem_avail = gpu_mem - min_gpu_mem_buffer
        if mem_avail < 0:
            continue

        num_workers = min(max_worker_per_gpu,
                          int(math.floor(mem_avail / min_total_worker_gpu_mem)))

        mem_per_worker = int((mem_avail / num_workers) -
                             worker_gpu_mem_overhead)

        for _ in range(num_workers):
            cores_avail_in_node = None
            next_node = node
            while True:
                next_node = (next_node + 1) % len(cores_avaliable)
                cores_avail_in_node = cores_avaliable[next_node]
                if len(cores_avail_in_node) >= cores_per_gpu_worker or next_node == node:
                    break
            if len(cores_avail_in_node) < cores_per_gpu_worker:
                break
            node = next_node

            worker_cores = [
                cores_avail_in_node.pop(-1) for i in range(cores_per_gpu_worker)]
            gpu_worker_configs.append(
                [[node], worker_cores, gpu_number, 1, mem_per_worker])

    cpu_worker_configs = []
    for node, cores_avail_in_node in enumerate(cores_avaliable):
        while True:
            cores_remaining = len(cores_avail_in_node)
            if cores_remaining < min_cores_per_cpu_worker:
                break

            num_workers = \
                max(1, int(round(cores_remaining / target_cores_per_cpu_worker)))
            num_cores = int(round(cores_remaining / num_workers))
            if cores_remaining < (num_cores + min_cores_per_cpu_worker):
                num_cores = cores_remaining

            worker_cores = [
                cores_avail_in_node.pop(-1) for i in range(num_cores)]
            cpu_worker_configs.append([[node], worker_cores, 0, 0, 0])

    gpu_run_script = get_run_script(
        'gpu_run_script.sh', 'custom_gpu_run_script.sh')
    cpu_run_script = get_run_script(
        'cpu_run_script.sh', 'custom_cpu_run_script.sh')

    workers = []
    for config in gpu_worker_configs:
        workers.append(run_worker(gpu_run_script, project, queue,
                       workers, config))

    for config in cpu_worker_configs:
        workers.append(run_worker(cpu_run_script, project, queue,
                       workers, config))

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
