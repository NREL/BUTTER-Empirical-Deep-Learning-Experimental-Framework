"""
Enqueues jobs from stdin into the JobQueue
"""

import argparse

import jobqueue
import sys
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('project', help='project identifier in your jobqueue.json file')
    parser.add_argument('group', help='group name or tag')
    args = parser.parse_args()

    jq = jobqueue.JobQueue(args.project, args.group)

    for line in sys.stdin.readlines():
        print(line)
        jq.add_job(json.loads(line))
