import argparse
import json
import platform
import random
import select
import subprocess
import sys
import time
from queue import Queue

import multiprocessing
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('worker', help='python worker module')
    parser.add_argument(
        'project', help='project identifier in your jobqueue.json file')
    parser.add_argument('queue', help='queue id')
    parser.add_argument('worker_configs',
                        help='A json list of worker configs. Example: [[0,2,0,1,4096], [2,34,0,0,0], [34,36,1,2,4096]] places three workers:  one on CPUs 0-1 and GPU 0, one on CPUs 2-33, and one on CPUs 34-35 and GPU 1.'
                        )
    args = parser.parse_args()

    host = platform.node()

    configs = json.loads(args.worker_configs)

    if not isinstance(configs[0], int):
        num_cpu = multiprocessing.cpu_count()
        num_gpu = len(get_available_gpus())
        if num_gpu == 2:
            configs = [[0, 1, 0, 1, 4228], [1, 2, 0, 1, 4228], [2, 3, 0, 1, 4228], [
                3, 4, 1, 2, 4228], [4, 5, 1, 2, 4228], [5, 6, 1, 2, 4228]]
        elif num_gpu == 1:
            configs = [[0, 1, 0, 1, 6000], [1, 2, 0, 1, 6000], [
                2, 3, 0, 1, 6000], [3, 4, 0, 1, 6000], [4, 5, 0, 1, 6000]]

        base = len(configs)
        cpu_per_worker = 2
        num_cpu_workers = int((num_cpu-base)/cpu_per_worker)
        for i in range(num_cpu_workers):
            configs.append([base + i*cpu_per_worker, base +
                           (i+1)*cpu_per_worker, 0, 0, 0])
    else:
        configs = configs[1:]

    print(
        f'Started Node Manager on host "{host}" for project "{args.project}" and queue "{args.queue}".')
    print(f'Launching worker processes...')
    print(json.dumps(configs))

    workers = []
    for rank, config in enumerate(configs):
        command = ['python', '-u', '-m', 'dmp.jobqueue_interface.worker_manager',
                   'python', '-u', '-m', 'dmp.jobqueue_interface.worker',
                   *[str(e) for e in config], args.project, args.queue]
        print(
            f'Creating subprocess {rank} with command: "{" ".join(command)}"')
        worker = subprocess.Popen(
            command, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            close_fds=True)
        workers.append(worker)
        # wait a bit to avoid overwhelming the database, etc...
        # time.sleep(random.uniform(1, 10))

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
