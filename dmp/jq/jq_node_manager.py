import argparse
import json
import platform
import select
import subprocess
from queue import Queue, Empty

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('worker', help='python worker module')
    parser.add_argument('project', help='project identifier in your jobqueue.json file')
    parser.add_argument('group', help='group name or tag')
    parser.add_argument('worker_configs',
                        help='A json list of worker configs. Example: [[0,2,0,1,4096], [2,34,0,0,0], [34,36,1,2,4096]] places three workers:  one on CPUs 0-1 and GPU 0, one on CPUs 2-33, and one on CPUs 34-35 and GPU 1.'
                        )
    args = parser.parse_args()

    host = platform.node()

    configs = json.loads(args.worker_configs)
    # print(
    #     f'Started Node Manager on host "{host}" for project "{args.project}" and group "{args.group}" executing worker module "{args.worker}".')
    print(f'Started Node Manager on host "{host}" for project "{args.project}" and group "{args.group}".')
    print(f'Launching worker processes...')
    print(json.dumps(configs))

    if __name__ == '__main__':
        logfile = open('logfile.txt', 'w')

        q = Queue()
        workers = []
        for rank, config in enumerate(configs):
            command = ['python', '-m', 'dmp.jq.jq_worker_manager',
                       'python', '-m', 'dmp.jq.jq_worker',
                       *[str(e) for e in config], args.project, args.group]
            print(f'Creating subprocess {rank} with command: "{" ".join(command)}"')
            worker = subprocess.Popen(
                command, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            workers.append(worker)

        streams = [w.stdout for w in workers]
        stream_name_map = {id(s): f'{i}:' for i, s in enumerate(streams)}


        def output(stream, line):
            if line == '':
                return
            name = stream_name_map[id(stream)]
            line = name + line
            print(line)
            # sys.stdout.write(line)
            # sys.stdout.flush()


        print('Starting output redirection...')
        while True:
            rstreams, _, _ = select.select(streams, [], [])
            for stream in rstreams:
                line = stream.readline()
                if line != '':
                    output(stream, line)
            if all(w.poll() is not None for w in workers):
                break

        for stream in streams:
            while True:
                line = stream.readline()
                if line == '':
                    break
                output(stream, line)

        print(f'Waiting for worker processes to exit...')
        for worker in workers:
            worker.wait()
        print('Exiting Worker Manager...')
