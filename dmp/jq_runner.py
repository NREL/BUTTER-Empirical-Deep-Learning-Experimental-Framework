import subprocess
import time
import jobqueue
import sys
import json

if __name__ == "__main__":

    jq = jobqueue.JobQueue(sys.argv[1], sys.argv[2])

    while True:
        time.sleep(0.1)
        message = jq.get_message()
        if message is None:
            continue
        try:
            print(f"{message.uuid} RUNNING")
            config_module = message.config["jq_module"]
            message.config["jq_uuid"] = message.uuid
            config_str = json.dumps(message.config)
            result = subprocess.run(["python", "-u", "-m", config_module, config_str], capture_output=True)
            if result.returncode != 0:
                print(f"{message.uuid} exited with bad retun code: {result.returncode}")
                print(f"-- stdout: {result.stdout}")
                print(f"-- stderr: {result.stderr}")
            else:
                message.mark_complete()
                print(f"{message.uuid} DONE")
        except Exception as e:
            print(f"Unknown exception in jq_runner: {e}")


# SELECT * FROM jobqueue WHERE groupname='test_tag';
# DELETE FROM jobqueue WHERE groupname='test_tag';
