"""
*** unused
jqrunner
Launches subprocesses that each run a single DMP experiment.
"""

import subprocess
import time
import jobqueue
import sys
import json

if __name__ == "__main__":

    jq = jobqueue.JobQueue(sys.argv[1], sys.argv[2])

    while True:
        message = jq.get_message()
        if message is None:
            time.sleep(5)
            continue
        try:
            print(f"Job Queue: {message.uuid} RUNNING")
            config_module = message.config["jq_module"]
            message.config["jq_uuid"] = message.uuid
            config_str = json.dumps(message.config)
            result = subprocess.run(["python", "-u", "-m", config_module, config_str])
            if result.returncode != 0:
                print(f"Job Queue: {message.uuid} exited with bad retun code: {result.returncode}")
            else:
                message.mark_complete()
                print(f"Job Queue: {message.uuid} DONE")
        except Exception as e:
            print(f"Job Queue: Unknown exception in jq_runner: {e}")


