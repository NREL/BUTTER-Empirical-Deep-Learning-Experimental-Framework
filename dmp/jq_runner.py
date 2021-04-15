import subprocess
import time
import jobqueue
import sys
import json

if __name__ == "__main__":

    jq = jobqueue.JobQueue(sys.argv[1], sys.argv[2])
    cmd = sys.argv[3]

    print(cmd)

    while True:
        time.sleep(0.1)
        message = jq.get_message()
        if message is None:
            continue
        try:
            result = subprocess.run(["python", "-u", "-m", cmd, json.dumps(message.config)])
            message.mark_complete()
        except Exception as e:
            print(e)


# SELECT * FROM jobqueue WHERE groupname='test_tag';
# DELETE FROM jobqueue WHERE groupname='test_tag';
