"""
jq slurm
Launches slurm jobs to finish a queue
"""

import subprocess
import time
import jobqueue
import sys
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="project identifier in your jobqueue.json file")
    parser.add_argument("tag", help="tag or group name")
    args = parser.parse_args()

    print("SLURM NANNY ACTIVATED")
    print("Nannying batch jobs for {} {}".format(args.project, args.tag))

    jq = jobqueue.JobQueue(args.project, args.tag)

    while True:

        messages = jq.messages
        print("Jobs remaining: {}".format(messgaes))
        if messages == 0:
            break

        print("Starting new SLURM Job.")
        result = subprocess.call(["sbatch", "--wait", "squeuebatchrunner.sh", args.project, args.tag])
        
        # blocks until slurm job is done

        jq.reset_incomplete_jobs()

    print("Done.")





