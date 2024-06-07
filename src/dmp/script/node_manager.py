import math
import platform
import re
import select
import sys
import subprocess
import os
from typing import IO, Any, Dict, List, Tuple
from dataclasses import dataclass
from numpy import append


def get_run_script(default_script, custom_script):
    if os.path.exists(custom_script):
        return custom_script
    return default_script


def make_worker_process(rank: int, command: list) -> subprocess.Popen:
    command = [str(a) for a in command]
    print(f'Creating subprocess {rank} with command: "{" ".join(command)}"')
    return subprocess.Popen(
        command,
        bufsize=1,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )


@dataclass
class WorkerConfig:
    run_script: str
    cpus: list
    gpus: list
    gpu_memory: int


def run_worker(
    rank: int,
    project: str,
    queue: int,
    config: WorkerConfig,
) -> subprocess.Popen:
    nodes = sorted({cpu[-3] for cpu in config.cpus})
    cpu_numbers = [cpu[-1] for cpu in config.cpus]

    cpus_string = ",".join([str(i) for i in cpu_numbers])
    nodes_string = ",".join([str(i) for i in nodes])
    gpus_string = (
        "-" if len(config.gpus) == 0 else ",".join([str(i) for i in config.gpus])
    )

    print(
        f'making worker command with nodes_string "{nodes_string}", cpus_string "{cpus_string}", gpus_string {gpus_string}".'
    )
    command = [
        f"./{config.run_script}",
        nodes_string,
        cpus_string,
        "python",
        "-u",
        "-m",
        "dmp.script.worker_manager",
        "python",
        "-u",
        "-m",
        "dmp.script.worker",
        project,
        queue,
        nodes_string,
        cpus_string,
        gpus_string,
        str(config.gpu_memory),
    ]

    return make_worker_process(rank, command)


def main():
    args = sys.argv
    project = args[1]
    queue = int(args[2])

    min_gpu_mem_per_worker = 12 * 1024
    worker_gpu_mem_overhead = 1024
    min_total_worker_gpu_mem = min_gpu_mem_per_worker + worker_gpu_mem_overhead

    min_gpu_mem_buffer = 500
    max_worker_per_gpu = 2

    cpus_per_gpu_worker = 4
    min_cpus_per_cpu_worker = 128

    smt_level = 1  # maximum number of SMT's (a.k.a. "CPUs") per core to use

    host = platform.node()

    print(
        f'Started Node Manager on host "{host}" for project "{project}" and queue "{queue}".'
    )
    print(f"Launching worker processes...")

    # num_cores = int(subprocess.check_output(
    #     'grep -c processor /proc/cpuinfo', shell=True))
    # num_sockets = int(subprocess.check_output(
    #     'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))

    # numa_hardware_output = subprocess.check_output(
    #     'numactl --hardware | grep -P "node \d+ cpus:"', shell=True).decode('ascii')

    # numa_nodes = [s for s in numa_hardware_output.split(
    #     '\n') if s.startswith('node ')]

    numa_physcpubind_output = subprocess.check_output(
        'numactl --show | grep -P "physcpubind"'
        # shell=True
    ).decode("ascii")
    avaliable_cpus = {
        int(i)
        for i in [
            i.replace("\n", "").strip()
            for i in numa_physcpubind_output[len("physcpubind: ") :].split(" ")
        ]
        if len(i) > 0
    }
    print(f"Available CPUs: {avaliable_cpus}")

    def get_or_add(d: Dict[Any, Dict], k) -> Dict:
        if k in d:
            return d[k]
        r = {}
        d[k] = r
        return r

    topology_depth = 4
    topology_index = {}
    lscpu_output = (
        subprocess.check_output("lscpu --all --parse", shell=True)
        .decode("ascii")
        .split("\n")
    )
    for line in lscpu_output:
        if len(line) >= 8 and line[0].isdigit():
            cols = [
                0 if len(e) == 0 else int(e) for e in line.split(",")[0:topology_depth]
            ]
            cpu, core, socket, node = cols
            get_or_add(get_or_add(get_or_add(topology_index, socket), node), core)[
                cpu
            ] = (socket, node, core, cpu)

    print(f"Detected CPU topology: {topology_index}")

    def build_cpu_index(
        level_indicies: List,
        level_dict: Dict,
    ) -> Tuple[int, List]:
        if len(level_indicies) + 1 >= topology_depth:
            basic_cpu_list = [
                cpu_index
                for cpu, cpu_index in level_dict.items()
                if cpu in avaliable_cpus
            ]
            basic_cpu_list.sort(reverse=True)
            result = [(1, cpu_index) for cpu_index in basic_cpu_list[-smt_level:]]
            return len(result), result

        result = []
        acc = 0
        level_items = sorted(level_dict.items(), reverse=True)
        level_indicies.append(None)
        for group, group_dict in level_items:
            level_indicies[-1] = group
            result.append(build_cpu_index(level_indicies, group_dict))
            acc += result[-1][0]
        level_indicies.pop()
        return acc, result

    num_cpus, top_level_cpu_index = build_cpu_index([], topology_index)

    print(f"Usable CPUs: {(num_cpus, top_level_cpu_index)}")

    def allocate_cpus(max_num_groups, min_group_size):
        nonlocal num_cpus, top_level_cpu_index

        def allocate_group():
            nonlocal num_cpus, top_level_cpu_index

            allocated = []

            def do_allocate_group(size, level):
                nonlocal allocated
                if isinstance(level, list):
                    # recursive case: allocate from largest sublevel
                    while len(level) > 0 and (
                        len(allocated) < min_group_size
                        or (len(groups) < max_num_groups and size < min_group_size)
                    ):
                        sublevel_size, sublevel = level.pop()
                        new_sublevel_size, new_sublevel = do_allocate_group(
                            sublevel_size, sublevel
                        )
                        if new_sublevel_size > 0:
                            # could be more efficient here
                            level.append((new_sublevel_size, new_sublevel))
                            level.sort(key=lambda e: e[0])
                        size -= sublevel_size - new_sublevel_size

                    return size, level
                else:
                    # base case: allocate this cpu
                    allocated.append(level)
                    return 0, None

            num_cpus, top_level_cpu_index = do_allocate_group(
                num_cpus, top_level_cpu_index
            )
            return allocated

        groups = []
        while len(groups) < max_num_groups and num_cpus > 0:
            groups.append(allocate_group())

        return groups

    # allocate GPU workers
    gpu_run_script = get_run_script("gpu_run_script.sh", "custom_gpu_run_script.sh")
    gpu_mems = []
    try:
        gpu_mems = [
            int(i)
            for i in subprocess.check_output(
                "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader",
                shell=True,
            ).splitlines()
        ]
    except:
        print("No GPUs detected using nvidia-smi.")

    worker_configs: List[WorkerConfig] = []
    for gpu_number, gpu_mem in enumerate(gpu_mems):
        mem_avail = gpu_mem - min_gpu_mem_buffer
        if mem_avail < 0:
            print(f"No GPU memory free for GPU {gpu_number}.")
            continue

        num_workers = min(
            max_worker_per_gpu, int(math.floor(mem_avail / min_total_worker_gpu_mem))
        )

        mem_per_worker = int((mem_avail / num_workers) - worker_gpu_mem_overhead)

        print(
            f"Allocating {num_workers} workers to GPU {gpu_number} with {mem_per_worker} MB GPU memory each."
        )
        cpu_groups = allocate_cpus(num_workers, cpus_per_gpu_worker)
        print(f"GPU {gpu_number} worker groups: {cpu_groups}")
        for cpu_group in cpu_groups:
            worker_configs.append(
                WorkerConfig(gpu_run_script, cpu_group, [gpu_number], mem_per_worker)
            )

    # allocate CPU workers
    cpu_run_script = get_run_script("cpu_run_script.sh", "custom_cpu_run_script.sh")
    cpu_groups = allocate_cpus(1000000000, min_cpus_per_cpu_worker)
    print(f"CPU groups: {cpu_groups}")
    worker_configs.extend(
        (WorkerConfig(cpu_run_script, cpu_group, [], 0) for cpu_group in cpu_groups)
    )

    # start workers
    workers = [
        run_worker(
            i,
            project,
            queue,
            config,
        )
        for i, config in enumerate(worker_configs)
    ]

    streams: List[IO[str]] = [w for w in [w.stdout for w in workers] if w is not None]
    stream_name_map = {id(s): f"{i}:" for i, s in enumerate(streams)}

    def output(stream, line):
        if len(line) == 0:
            return
        name = stream_name_map[id(stream)]
        if not isinstance(line, str):
            line = line.decode("utf-8")
        line = name + line
        sys.stdout.write(line)
        sys.stdout.flush()

    print("Starting output redirection...")
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

    print(f"Waiting for worker processes to exit...")
    for worker in workers:
        worker.wait()
    print("Exiting Worker Manager...")


if __name__ == "__main__":
    main()
