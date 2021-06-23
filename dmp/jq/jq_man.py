"""
jq man
Utility to manage a job queue
"""

import subprocess
import time
import jobqueue
import sys
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('project', help='project identifier in your jobqueue.json file')
    parser.add_argument('group', help='group name or tag')
    parser.add_argument('command', help='queue command: version, create, recreate, clear, ')
    args = parser.parse_args()

    print('SLURM NANNY ACTIVATED')
    print('Nannying batch jobs for {} {}'.format(args.project, args.tag))

    jq = jobqueue.JobQueue(args.project, args.tag)

    while True:

        # if arg.re_enqueue_failed_jobs:
        # jq.reset_incomplete_jobs() # We could remove this. If it's left in, jobs that take longer than the slurm max time will never finish.
                                   # jobs that fail many times
                                   # jobs that hang
        messages = jq.messages
        print('Jobs remaining: {}'.format(messages))
        if messages == 0:
            break

        print('Starting new SLURM Job.')
        result = subprocess.call(['sbatch', '--wait', 'slurm_job_runner.sh', args.project, args.tag])
        # blocks until slurm job is done

    print('Done.')





