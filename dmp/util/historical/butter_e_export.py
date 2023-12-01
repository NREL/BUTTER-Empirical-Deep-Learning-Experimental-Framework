import numpy
import pandas
import io
import uuid
import psycopg.sql
import pyarrow
import pyarrow.parquet

import jobqueue
from jobqueue.connection_manager import ConnectionManager


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

import dmp.keras_interface.model_serialization as model_serialization
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.table import Table
from dmp.postgres_interface.element.column_group import ColumnGroup

pd.options.display.max_seq_items = None


class Run(Table):
    experiment_id: Column = Column("experiment_id", "uuid")
    run_timestamp: Column = Column("run_timestamp", "timestamp")
    run_id: Column = Column("run_id", "uuid")
    job_id: Column = Column("job_id", "uuid")
    seed: Column = Column("seed", "bigint")
    slurm_job_id: Column = Column("slurm_job_id", "bigint")
    task_version: Column = Column("task_version", "smallint")
    num_nodes: Column = Column("num_nodes", "smallint")
    num_cpus: Column = Column("num_cpus", "smallint")
    num_gpus: Column = Column("num_gpus", "smallint")
    gpu_memory: Column = Column("gpu_memory", "integer")
    host_name: Column = Column("host_name", "text")
    batch: Column = Column("batch", "text")
    run_data: Column = Column("run_data", "jsonb")
    run_history: Column = Column("run_history", "bytea")
    run_extended_history: Column = Column("run_extended_history", "bytea")


run = Run("run")



class JobStatus(Table):
    queue: Column = Column("queue", "smallint")
    status: Column = Column("status", "smallint")
    priority: Column = Column("priority", "integer")
    id: Column = Column("id", "uuid")
    start_time: Column = Column("start_time", "timestamp")
    update_time: Column = Column("update_time", "timestamp")
    worker: Column = Column("worker", "uuid")
    error_count: Column = Column("error_count", "smallint")
    error: Column = Column("error", "text")
    parent: Column = Column("parent", "uuid")


job_status = JobStatus("job_status")


class JobData(Table):
    id: Column = Column("id", "uuid")
    command: Column = Column("command", "jsonb")


job_data = JobData("job_data")


