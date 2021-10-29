import argparse
import json
import platform
import subprocess
import sys
from asyncio import Queue, QueueEmpty
from queue import Empty
from threading import Thread

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


    def enqueue_output(out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()


    if __name__ == '__main__':
        logfile = open('logfile.txt', 'w')

        workers = []
        for rank, config in enumerate(configs):
            command = ['python', '-m', 'dmp.jq.jq_worker_manager',
                       'python', '-m', 'dmp.jq.jq_worker',
                       *[str(e) for e in config], args.project, args.group]
            print(f'Creating subprocess with command: "{" ".join(command)}"')
            workers.append(subprocess.Popen(
                command, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

        q = Queue()
        threads = []
        for w in workers:
            t = Thread(target=enqueue_output, args=(w.stdout, q))
            threads.append(t)
            t.daemon = True
            t.start()

        while True:
            try:
                line = q.get_nowait()
            except QueueEmpty:
                pass
            else:
                sys.stdout.write(line)
                logfile.write(line)
                logfile.flush()

            # break when all processes are done.
            if all(w.poll() is not None for w in workers):
                break

        print(f'Waiting for worker processes to exit...')
        for worker in workers:
            worker.wait()
        print('Exiting Worker Manager...')
