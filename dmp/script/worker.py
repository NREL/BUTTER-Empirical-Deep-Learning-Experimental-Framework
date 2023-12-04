import sys
import uuid

from jobqueue.job_queue import JobQueue
from jobqueue import load_credentials
from dmp import common

from dmp.postgres_interface.schema.postgres_interface import PostgresInterface

import tensorflow

from dataclasses import dataclass
import random
import time
import traceback
from typing import Any, Dict, List, Optional
import uuid
import tensorflow
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue
from dmp import common

from typing import TYPE_CHECKING
from dmp.run_entry import RunEntry

from dmp.task.run_status import RunStatus

if TYPE_CHECKING:
    from dmp.postgres_interface.schema.postgres_interface import PostgresInterface


# from .common import jobqueue_marshal


@dataclass
class Worker:
    _database: "PostgresInterface"
    # _result_logger: "ExperimentResultLogger"
    _strategy: tensorflow.distribute.Strategy
    _info: Dict[str, Any]
    _max_jobs: Optional[int] = None

    @property
    def strategy(self) -> tensorflow.distribute.Strategy:
        return self._strategy

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    @property
    def database(self) -> "PostgresInterface":
        return self._database

    def work_loop(
        self,
        queue_id: int,
        wait_until_exit=15 * 60,
        maximum_waiting_time=5 * 60,
    ) -> None:
        from dmp.marshaling import marshal
        from dmp.task.run_command import RunCommand

        # from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
        from dmp.context import Context

        print(f"Work Loop: Starting...", flush=True)

        git_hash = common.get_git_hash()

        wait_start = None
        wait_bound = 1.0

        while True:
            # Pull job off the queue
            runs = self.database.pop_runs(queue_id, worker_id=worker_id)

            if len(runs) <= 0:
                if wait_start is None:
                    wait_start = time.time()
                    wait_bound = 1
                else:
                    waiting_time = time.time() - wait_start
                    if waiting_time > wait_until_exit:
                        print(
                            "Work Loop: No runs found and max waiting time exceeded. Exiting...",
                            flush=True,
                        )
                        break

                # No runs, wait and try again.
                print("Work Loop No runs found. Waiting...", flush=True)

                # bounded randomized exponential backoff
                wait_bound = min(maximum_waiting_time, wait_bound * 2)
                time.sleep(random.uniform(1.0, wait_bound))
                continue

            run: RunEntry = runs[0]

            try:
                wait_start = None

                print(f"Work Loop: {run.id} running...", flush=True)

                """
                + save resume task
                + save history
                + save model
                ---
                + load resume task
                + resume from saved history
                + resume from saved model
                """

                second_git_hash = common.get_git_hash()
                if second_git_hash is not git_hash and second_git_hash != git_hash:
                    break

                self._info["worker_id"] = worker_id

                # demarshal task from job.command
                run_command: RunCommand = run.command

                # run task
                with self.strategy.scope():
                    run_command(Context(self, run))

                # log task run
                # if isinstance(result, ExperimentResultRecord):
                #     self._result_logger.log(result)

                if self._max_jobs is not None:
                    self._max_jobs -= 1

                second_git_hash = common.get_git_hash()
                if (self._max_jobs is not None and self._max_jobs <= 0) or (
                    second_git_hash is not None and git_hash != second_git_hash
                ):
                    break

                # Mark the job as complete in the queue.
                self.database.set_status(run, RunStatus.Complete)

                print(f"Work Loop: {run.id} done.", flush=True)
            except Exception as e:
                print(
                    f"Work Loop: {run.id} unhandled exception {e} in work_loop.",
                    flush=True,
                )
                print(traceback.format_exc())
                try:
                    run.error_message = str(e) + "\n" + traceback.format_exc()
                    run.status = RunStatus.Failed
                    self.database.update_runs((run,))
                except Exception as e2:
                    print(
                        f"Work Loop: {run.id} exception thrown while marking as failed in work_loop: {e}, {e2}!",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
        print(f"Work Loop: exiting.", flush=True)


def make_strategy(num_cores, gpus, gpu_mem):
    if num_cores is None and gpus is None and gpu_mem is None:
        return tensorflow.distribute.get_strategy()

    if num_cores is None:
        import multiprocessing

        num_cores = max(1, multiprocessing.cpu_count() - 1)

    if gpus is None:
        gpus = []

    if gpu_mem is None:
        gpu_mem = 4096

    tf_gpus = tensorflow.config.list_physical_devices("GPU")
    print(f"Found GPUs: {len(tf_gpus)} {tf_gpus}.\nUsing: {gpus}.")
    gpu_set = set(gpus)
    gpu_devices = []
    for gpu in tf_gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
        number = int(gpu.name.split(":")[-1])
        if number in gpu_set:
            gpu_devices.append(gpu)

            # tensorflow.config.experimental.set_virtual_device_configuration(
            #     gpu, [

            #         tensorflow.config.experimental.VirtualDeviceConfiguration(
            #             memory_limit=gpu_mem)
            #     ])

    cpus = tensorflow.config.list_physical_devices("CPU")
    # print(f'cpus: {cpus}')
    visible_devices = gpu_devices.copy()
    visible_devices.extend(cpus)
    tensorflow.config.set_visible_devices(visible_devices)
    tensorflow.config.set_soft_device_placement(True)

    tensorflow.config.threading.set_intra_op_parallelism_threads(num_cores)
    tensorflow.config.threading.set_inter_op_parallelism_threads(num_cores)

    if len(gpu_devices) > 1:
        print(visible_devices)
        print(gpu_devices)
        strategy = tensorflow.distribute.MirroredStrategy(
            devices=[d.name[len("/physical_device:") :] for d in gpu_devices]
        )
        #   cross_device_ops=tensorflow.contrib.distribute.AllReduceCrossDeviceOps(
        #      all_reduce_alg="hierarchical_copy")
    else:
        strategy = tensorflow.distribute.get_strategy()
    return strategy


# Example:
# python -u -m dmp.jobqueue_interface.worker dmp 11 '0' '0,1,2,3,4,5,6,7,8,9,10,11,12' '0' 4096
# python -u -m dmp.jobqueue_interface.worker dmp 10 '0,1' '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35' '0,1' 15360
if __name__ == "__main__":
    a = sys.argv
    print(a)

    database = a[1]
    queue_id = int(a[2])

    nodes = [int(e) for e in a[3].split(",")]
    cpus = [int(e) for e in a[4].split(",")]
    gpus = (
        [int(e) for e in a[5].split(",")]
        if len(a) > 5 and len(a[5]) > 0 and a[5] != "-"
        else []
    )
    gpu_memory = int(a[6]) if len(a) > 6 and len(a[6]) > 0 else 0

    tensorflow.keras.backend.set_floatx("float32")

    worker_id = uuid.uuid4()
    print(f"Worker id {worker_id} starting...")
    print("\n", flush=True)

    if not isinstance(queue_id, int):
        queue_id = 1

    print(f"Worker id {worker_id} load credentials...\n", flush=True)
    credentials = load_credentials(database)
    print(f"Worker id {worker_id} initialize database schema...\n", flush=True)
    schema = PostgresInterface(credentials)
    print(f"Worker id {worker_id} create job queue...\n", flush=True)
    job_queue = JobQueue(credentials, int(queue_id), check_table=False)
    # print(f"Worker id {worker_id} create result logger..\n", flush=True)
    # result_logger = PostgresCompressedResultLogger(schema)
    print(f"Worker id {worker_id} create Worker object..\n", flush=True)

    strategy = make_strategy(
        len(cpus),
        gpus,
        gpu_memory,
    )

    worker = Worker(
        schema,
        strategy,
        {
            "nodes": nodes,
            "cpus": cpus,
            "gpus": gpus,
            "num_cpus": len(cpus),
            "num_gpus": len(gpus),
            "num_nodes": len(nodes),
            "gpu_memory": gpu_memory,
            "tensorflow_strategy": str(type(strategy)),
            "queue_id": queue_id,
        },
    )
    print(f"Worker id {worker_id} start Worker object...\n", flush=True)
    worker.work_loop(queue_id)  # runs the work loop on the worker
    print(f"Worker id {worker_id} Worker exited.\n", flush=True)
