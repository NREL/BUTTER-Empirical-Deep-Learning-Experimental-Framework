import dmp.experiment.aspect_test as exp
from dmp.data.logging import write_log
import time
import jobqueue
import sys
import json

if __name__ == "__main__":

    jq = jobqueue.JobQueue(sys.argv[1], sys.argv[2])

    while True:
        message = jq.get_message()
        if message is None:
            time.sleep(1)
            continue
        try:
            print(f"Job Queue: {message.uuid} RUNNING")

            result = exp.aspect_test(message.config)
            
            write_log(result, message.config["log"],
                      name = result["run_name"],
                      job = message.uuid)

            message.mark_complete()

            print(f"Job Queue: {message.uuid} DONE")
        except Exception as e:
            print(f"Job Queue: {message.uuid} Unknown exception in jq_runner: {e}")


# SELECT * FROM jobqueue WHERE groupname='test_tag';
# DELETE FROM jobqueue WHERE groupname='test_tag';
