import dmp.experiment.aspect_test as exp
from dmp.data.logging import write_log
import tensorflow as tf
import time
import jobqueue
import sys
import os

def start_jobqueue():

    jq = jobqueue.JobQueue(sys.argv[1], sys.argv[2])

    while True:
        
        # Pull job off the queue
        message = jq.get_message()

        if message is None:
            # No jobs, wait one second and try again.
            time.sleep(1)
            continue
        try:
            print(f"Job Queue: {message.uuid} RUNNING")

            # Run the experiment
            result = exp.aspect_test(message.config)

            # Write the log
            write_log(result, message.config["log"],
                      name = result["run_name"],
                      job = message.uuid)

            # Mark the job as complete in the queue.
            message.mark_complete()

            print(f"Job Queue: {message.uuid} DONE")
        except Exception as e:
            print(f"Job Queue: {message.uuid} Unknown exception in jq_runner: {e}")


def tf_gpu_session():
    """
    Return a tensorflow session with the configuration we'll use on a GPU node.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    return tf.Session(config=config)


def start_jobqueue_with_device_based_on_type(DMP_TYPE, DMP_RANK):
    
    devices = tf.config.list_physical_devices(device_type=DMP_TYPE)
    device_name = devices[DMP_RANK%len(devices)].name
    
    with tf.device(device_name):
        
        if type=="GPU":
            sess = tf_gpu_session()

        print(f"Starting {DMP_TYPE} job queue on device {device_name}")
        start_jobqueue()


if __name__ == "__main__":

    DMP_TYPE = os.getenv('DMP_TYPE')
    DMP_RANK = os.getenv('DMP_RANK')

    if DMP_TYPE is not None and DMP_RANK is not None:
        start_jobqueue_with_device_based_on_type(DMP_TYPE, int(DMP_RANK))
    else:
        print("Starting job queue without explicit device placement")
        start_jobqueue()

