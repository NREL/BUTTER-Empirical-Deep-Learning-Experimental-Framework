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
            print("No Jobs, waiting")
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
    config.gpu_options.allow_growth=False
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    return tf.Session(config=config)

## Set number of threads
# session_conf = tf.ConfigProto(
#       intra_op_parallelism_threads=1,
#       inter_op_parallelism_threads=1)
# sess = tf.Session(config=session_conf)

## Detect when more than one GPU with nvidia-smi

def start_jobqueue_with_device_based_on_type(DMP_TYPE, DMP_RANK):
    
    if type=="GPU":
        with tf.device("/device:GPU:0"):
            with tf_gpu_session().as_default():
                start_jobqueue()
        
    else:
        with tf.device("/device:CPU:0"):
            start_jobqueue()

# def start_jobqueue_with_device_based_on_type(DMP_TYPE, DMP_RANK):
    
#     devices = tf.config.list_logical_devices(device_type=DMP_TYPE)

#     device = devices[DMP_RANK%len(devices)]

#     print(f"Starting {DMP_TYPE} rank {DMP_RANK} job queue on device {device.name}")

#     with tf.device(device.name):
        
#         if type=="GPU":
#             with tf_gpu_session().as_default():
#                 start_jobqueue()
        
#         else:
#             start_jobqueue()


if __name__ == "__main__":

    DMP_TYPE = os.getenv('DMP_TYPE')
    DMP_RANK = os.getenv('DMP_RANK')

    if DMP_TYPE is not None and DMP_RANK is not None:
        start_jobqueue_with_device_based_on_type(DMP_TYPE, int(DMP_RANK))
    else:
        print("Starting job queue without explicit device placement")
        start_jobqueue()

