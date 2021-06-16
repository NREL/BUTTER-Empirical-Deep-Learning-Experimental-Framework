import argparse
import json
import platform
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('worker', help='python worker module')
    parser.add_argument('project', help='project identifier in your jobqueue.json file')
    parser.add_argument('group', help='group name or tag')
    parser.add_argument('worker_configs',
                        help='A json list of worker configs. Example: [[0,2,0,1], [2,34,0,0], [34,36,1,2]] places three workers:  one on CPUs 0-1 and GPU 0, one on CPUs 2-33, and one on CPUs 34-35 and GPU 1.'
                        )
    args = parser.parse_args()

    host = platform.node()

    configs = json.loads(args.worker_configs)
    # print(
    #     f'Started Node Manager on host "{host}" for project "{args.project}" and group "{args.group}" executing worker module "{args.worker}".')
    print(f'Started Node Manager on host "{host}" for project "{args.project}" and group "{args.group}".')
    print(f'Launching worker processes...')
    print(json.dumps(configs))
    workers = [subprocess.Popen(
        ['python', '-m', 'dmp.jq.jq_worker_manager',
         'python', '-m', 'dmp.jq.jq_worker',
         *[str(e) for e in config],
         args.project,
         args.group])
        for rank, config in enumerate(configs)]
    print(f'Waiting for worker processes to exit...')
    for worker in workers:
        worker.wait()
    print('Exiting Worker Manager...')
