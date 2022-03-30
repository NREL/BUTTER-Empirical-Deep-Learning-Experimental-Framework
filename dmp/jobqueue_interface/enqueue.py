"""
Enqueues jobs from stdin into the JobQueue
"""

import argparse

from jobqueue.connect import connect
from jobqueue.job_queue import JobQueue
from dmp.jobqueue_interface.common import jobqueue_marshal

import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'project', help='project identifier in your jobqueue.json file')
    parser.add_argument('queue', help='queue id to use (smallint)')
    # parser.add_argument('group', help='group name or tag')
    args = parser.parse_args()

    credentials = connect.load_credentials('dmp')
    job_queue = JobQueue(credentials, int(args.queue), check_table=False)

    # job_queue = jobqueue.JobQueue(args.project, args.group, pooling=True)

    accumulated = []
    lines = sys.stdin.readlines()

    for line in lines:
        task = jobqueue_marshal.demarshal(line)
        accumulated.append(task)
        if len(accumulated) > 512:
            job_queue.push(accumulated)
            accumulated.clear()
    job_queue.push(accumulated)
