import jobqueue
import sys
import json

if __name__ == "__main__":

    jq = jobqueue.JobQueue(sys.argv[1], sys.argv[2])

    for line in sys.stdin.readlines():
        print(line)
        jq.add_job(json.loads(line))

