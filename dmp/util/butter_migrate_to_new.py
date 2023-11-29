import io
import os
from itertools import chain
from psycopg.types.json import Jsonb

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from jobqueue.connection_manager import ConnectionManager
import argparse
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import pandas
import traceback
import json
import pyarrow

from pprint import pprint
import uuid
from psycopg.sql import SQL, Identifier, Literal

from jobqueue import load_credentials
from jobqueue.cursor_manager import CursorManager

from dmp import marshaling
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table
from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.task.experiment.experiment import Experiment
from dmp.uuid_tools import json_to_uuid

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)


import pathos.multiprocessing as multiprocessing

"""
update migration set status = 0 where TRUE
	AND status = 1
	AND (not exists (select 1 from history where history.id = migration.run_id)
	or not exists (select 1 from run_status where run_status.id = migration.run_id)
	or not exists (select 1 from run_data where run_data.id = migration.run_id));



SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE usename = 'dmpappsops'
    AND query_start < (CURRENT_TIMESTAMP - '10s'::interval)
ORDER BY state, query_start desc;


select *, (transfered*100.0/total) percent_complete
FROM (
select sum((status = 1)::integer) transfered, count(1) total from migration
	) x;


+ set old_experiment_id in experiment2 table
+ extract new experiment summary for butter runs
+ extract butter-e run dataset
"""

pmlb_index_path = os.path.join(
    os.path.realpath(
        os.path.join(
            os.getcwd(),
            os.path.dirname(__file__),
        )
    ),
    "pmlb.csv",
)
dataset_index = pandas.read_csv(pmlb_index_path)
dataset_index.set_index("Dataset", inplace=True, drop=False)

dataset_shape_map = {
    "201_pol": (26, 11),
    "294_satellite_image": (36, 6),
    "505_tecator": (124, 1),
    "529_pollen": (4, 1),
    "537_houses": (8, 1),
    "adult": (81, 1),
    "banana": (2, 1),
    "connect_4": (126, 3),
    "mnist": (784, 10),
    "nursery": (26, 4),
    "sleep": (141, 5),
    "splice": (287, 3),
    "wine_quality_white": (11, 7),
}


class JobStatusTable(Table):
    queue = Column("queue", "smallint")
    status = Column("status", "smallint")
    priority = Column("priority", "integer")
    id = Column("id", "uuid")
    start_time = Column("start_time", "timestamp")
    update_time = Column("update_time", "timestamp")
    worker = Column("worker", "uuid")
    error_count = Column("error_count", "smallint")
    error = Column("error", "text")
    parent = Column("parent", "uuid")


job_status_table: JobStatusTable = JobStatusTable("job_status")

job_status_selection: ColumnGroup = ColumnGroup(
    job_status_table.start_time,
    job_status_table.update_time,
    job_status_table.worker,
    job_status_table.queue,
    job_status_table.error_count,
    job_status_table.error,
    job_status_table.parent,
)


class JobDataTable(Table):
    id = Column("id", "uuid")
    command = Column("command", "jsonb")


job_data_table: JobDataTable = JobDataTable("job_data")

job_data_selection: ColumnGroup = ColumnGroup(job_data_table.command)


class ExperimentTable(Table):
    experiment_id = Column("experiment_id", "uuid")
    old_experiment_id = Column("old_experiment_id", "integer")
    experiment_attrs = Column("experiment_attrs", "integer[]")
    experiment_tags = Column("experiment_tags", "integer[]")


experiment_table: ExperimentTable = ExperimentTable("experiment")

experiment_selection: ColumnGroup = ColumnGroup(
    experiment_table.experiment_id,
    experiment_table.old_experiment_id,
)


class OldExperimentTable(Table):
    experiment_id = Column("experiment_id", "integer")
    experiment_parameters = Column("experiment_parameters", "smallint[]")
    num_free_parameters = Column("num_free_parameters", "bigint")
    widths = Column("widths", "integer[]")
    relative_size_error = Column("relative_size_error", "real")
    b_300_epoch_sweep = Column("300_epoch_sweep", "boolean")
    b_30k_epoch_sweep = Column("30k_epoch_sweep", "boolean")
    learning_rate_sweep = Column("learning_rate_sweep", "boolean")
    label_noise_sweep = Column("label_noise_sweep", "boolean")
    batch_size_sweep = Column("batch_size_sweep", "boolean")
    regularization_sweep = Column("regularization_sweep", "boolean")
    optimizer_sweep = Column("optimizer_sweep", "boolean")
    learning_rate_batch_size_sweep = Column("learning_rate_batch_size_sweep", "boolean")
    size_adjusted_regularization_sweep = Column(
        "size_adjusted_regularization_sweep", "boolean"
    )
    butter = Column("butter", "boolean")


old_experiment_table: OldExperimentTable = OldExperimentTable("experiment_")
old_experiment_selection: ColumnGroup = ColumnGroup(
    # old_experiment_table.experiment_id,  # "old experiment id"
    old_experiment_table.num_free_parameters,
    old_experiment_table.widths,
    old_experiment_table.relative_size_error,
    old_experiment_table.b_300_epoch_sweep,
    old_experiment_table.b_30k_epoch_sweep,
    old_experiment_table.learning_rate_sweep,
    old_experiment_table.label_noise_sweep,
    old_experiment_table.batch_size_sweep,
    old_experiment_table.regularization_sweep,
    old_experiment_table.optimizer_sweep,
    old_experiment_table.learning_rate_batch_size_sweep,
    old_experiment_table.size_adjusted_regularization_sweep,
    old_experiment_table.butter,
)


class RunTable(Table):
    experiment_id = Column("experiment_id", "uuid")
    run_timestamp = Column("run_timestamp", "timestamp")
    run_id = Column("run_id", "uuid")
    job_id = Column("job_id", "uuid")
    seed = Column("seed", "bigint")
    slurm_job_id = Column("slurm_job_id", "bigint")
    task_version = Column("task_version", "smallint")
    num_nodes = Column("num_nodes", "smallint")
    num_cpus = Column("num_cpus", "smallint")
    num_gpus = Column("num_gpus", "smallint")
    gpu_memory = Column("gpu_memory", "integer")
    host_name = Column("host_name", "text")
    batch = Column("batch", "text")
    run_data = Column("run_data", "jsonb")
    run_history = Column("run_history", "bytea")
    run_extended_history = Column("run_extended_history", "bytea")
    task = Column("task", "jsonb")


run_table: RunTable = RunTable("run")


class MigrationTable(Table):
    status = Column("status", "integer")
    experiment_id = Column("experiment_id", "uuid")
    run_id = Column("run_id", "uuid NOT NULL")


migration_table: MigrationTable = MigrationTable("migration")


selected_columns = (
    job_status_selection
    + job_data_selection
    + experiment_selection
    + old_experiment_selection
    + run_table
)


def parquet_to_dataframe(value):
    with io.BytesIO(value) as buffer:
        df = pyarrow.parquet.read_table(
            pyarrow.PythonFile(buffer, mode="r")
        ).to_pandas()
        df.sort_values(by=["epoch"], inplace=True)
        return df


def do_work(args):
    from dmp.marshaling import marshal

    # global schema, credentials

    worker_number, block_size = args

    credentials = load_credentials("dmp")
    schema = PostgresSchema(credentials)

    worker_id = str(worker_number) + str(uuid.uuid4())
    total_num_converted = 0
    total_num_excepted = 0
    print(f"Worker {worker_number} : {worker_id} started...")

    get_runs_query = SQL(
        """
WITH r AS (
SELECT
	{job_status_selection},
    {job_data_selection},
    {experiment_selection},
    {old_experiment_selection},
    {run_selection}
FROM
    (
        SELECT experiment_id, run_id
        FROM migration
        WHERE
            status = 0
        ORDER BY experiment_id, run_id
        FOR UPDATE
        SKIP LOCKED
        LIMIT {block_size}
    ) m,
    job_status s,
    job_data d,
    run r,
	experiment e,
    experiment_ eold
WHERE TRUE
	AND s.id = m.run_id
    AND d.id = s.id
    AND r.run_id = s.id
	AND e.experiment_id = m.experiment_id
    AND e.old_experiment_id = eold.experiment_id
),
updated AS (
    UPDATE migration
        SET status = 1
    WHERE
        migration.run_id IN (SELECT run_id from r)
)
SELECT * from r
;"""
    ).format(
        job_status_selection=job_status_selection.of(Identifier("s")),
        job_data_selection=job_data_selection.of(Identifier("d")),
        experiment_selection=experiment_selection.of(Identifier("e")),
        old_experiment_selection=old_experiment_selection.of(Identifier("eold")),
        run_selection=run_table.of(Identifier("r")),
        block_size=Literal(block_size),
    )

    while True:  #  binary=True, scrollable=True
        num_converted = 0
        num_excepted = 0

        source_records = []
        with ConnectionManager(credentials) as connection:
            connection.execute("SET TIMEZONE to 'UTC';")
            with connection.cursor(binary=True) as cursor:
                cursor.execute(get_runs_query, binary=True)
                source_records = list(cursor.fetchall())

        result_records = []
        experiment_ids: Dict[uuid.UUID, Experiment] = {}

        for row in source_records:

            def get_value(column: Column):
                return row[selected_columns[column]]

            run_id = get_value(run_table.run_id)
            job_id = get_value(run_table.job_id)

            try:
                src_command = get_value(job_data_table.command)
                src_run_data = get_value(run_table.run_data)
                # src_task = get_value(run_table.task)

                # src_history = parquet_to_dataframe(get_value(run_table.run_history))
                # src_history = src_history.join(
                #     parquet_to_dataframe(get_value(run_table.run_extended_history)),
                #     on="epoch",
                #     how="left",
                #     rsuffix="_",
                # )

                dataset_name = src_command["dataset"]["name"]
                dsinfo = dataset_index[dataset_index["Dataset"] == dataset_name].iloc[0]
                ml_task = MLTask.regression
                num_outputs = 1

                if dataset_name == "201_pol":
                    ml_task = MLTask.classification
                    num_outputs = 11
                elif dataset_name == "294_satellite_image":
                    ml_task = MLTask.classification
                    num_outputs = 6
                elif dsinfo["Task"] == "classification":
                    ml_task = MLTask.classification
                    num_outputs = int(dsinfo["n_classes"])
                    if num_outputs == 2:
                        num_outputs = 1

                dataset_size = int(dsinfo["n_observations"])
                test_split = float(src_command["dataset"]["test_split"])
                train_set_size = math.floor(dataset_size * test_split)
                test_set_size = dataset_size - train_set_size

                shapes = dataset_shape_map[dataset_name]

                input_shape = [
                    shapes[0],
                ]

                output_activation = "softmax"
                output_kernel_initializer = "GlorotUniform"
                loss = "CategoricalCrossentropy"
                if ml_task == MLTask.regression:
                    output_activation = "sigmoid"
                    output_kernel_initializer = "GlorotUniform"
                    loss = "MeanSquaredError"
                elif ml_task == MLTask.classification:
                    if num_outputs == 1:
                        output_activation = "sigmoid"
                        output_kernel_initializer = "GlorotUniform"
                        loss = "BinaryCrossentropy"
                    else:
                        output_activation = "softmax"
                        output_kernel_initializer = "GlorotUniform"
                        loss = "CategoricalCrossentropy"

                src_history = parquet_to_dataframe(get_value(run_table.run_history))
                dst_command = {
                    "run": {
                        "data": {
                            "job_id": {
                                "type": "UUID",
                                "value": str(run_id),
                            },
                            "run_id": {
                                "type": "UUID",
                                "value": str(job_id),
                            },
                            "pre_migration": {
                                "experiment_id": get_value(
                                    experiment_table.experiment_id
                                ),
                                "old_experiment_id": get_value(
                                    experiment_table.old_experiment_id
                                ),
                                "queue": get_value(job_status_table.queue),
                            },
                            "context": {
                                "cpus": src_run_data["cpus"],
                                "gpus": src_run_data["gpus"],
                                "nodes": src_run_data["nodes"],
                                "num_cpus": get_value(run_table.num_cpus),
                                "num_gpus": get_value(run_table.num_gpus),
                                "queue_id": get_value(job_status_table.queue),
                                "num_nodes": get_value(run_table.num_nodes),
                                "worker_id": {
                                    "type": "UUID",
                                    "value": str(get_value(job_status_table.worker)),
                                },
                                "gpu_memory": get_value(run_table.gpu_memory),
                                "tensorflow_strategy": src_run_data["strategy"],
                            },
                            "git_hash": src_run_data["git_hash"],
                            "platform": src_run_data["platform"],
                            "host_name": get_value(run_table.host_name),
                            "slurm_job_id": get_value(run_table.slurm_job_id),
                            "python_version": src_run_data["python_version"],
                            "tensorflow_version": src_run_data["tensorflow_version"],
                        },
                        "seed": get_value(run_table.seed),
                        "type": "RunSpec",
                        "model_saving": None,
                        "record_times": False,
                        "saved_models": [],
                        "resume_checkpoint": None,
                        "record_post_training_metrics": False,
                    },
                    "type": "Run",
                    "experiment": {
                        "fit": {
                            "epochs": int(src_history["epoch"].max()),
                            "batch_size": src_command["fit"]["batch_size"],
                        },
                        "data": {
                            "batch": src_command["batch"],
                            "ml_task": str(ml_task),
                            "butter_e": get_value(old_experiment_table.butter),
                            "input_shape": input_shape,
                            "output_shape": [num_outputs],
                            "data_set_size": dataset_size,
                            "test_set_size": test_set_size,
                            "train_set_size": train_set_size,
                            "network_description": {
                                "widths": get_value(old_experiment_table.widths)
                            },
                            "num_free_parameters": get_value(
                                old_experiment_table.num_free_parameters
                            ),
                            "validation_set_size": 0,
                        },
                        "loss": {"class": loss},
                        "type": "TrainingExperiment",
                        "model": {
                            "size": src_command["model"]["size"],
                            "type": src_command["model"]["type"],
                            "depth": src_command["model"]["depth"],
                            "inner": src_command["model"]["inner"],
                            "input": {
                                "name": "dmp_2",
                                "type": "Input",
                                "shape": input_shape,
                            },
                            "shape": src_command["model"]["shape"],
                            "output": {
                                "type": src_command["model"]["inner"]["type"],
                                "units": num_outputs,
                                "use_bias": True,
                                "activation": output_activation,
                                "bias_constraint": None,
                                "bias_initializer": "Zeros",
                                "bias_regularizer": src_command["model"]["inner"][
                                    "bias_regularizer"
                                ],
                                "kernel_constraint": src_command["model"]["inner"][
                                    "kernel_constraint"
                                ],
                                "kernel_initializer": {
                                    "class": output_kernel_initializer
                                },
                                "kernel_regularizer": src_command["model"]["inner"][
                                    "kernel_regularizer"
                                ],
                                "activity_regularizer": src_command["model"]["inner"][
                                    "activity_regularizer"
                                ],
                            },
                            "search_method": src_command["model"]["search_method"],
                        },
                        "dataset": {
                            "name": dataset_name,
                            "type": "DatasetSpec",
                            "method": src_command["dataset"]["method"],
                            "source": "pmlb",
                            "test_split": src_command["dataset"]["test_split"],
                            "label_noise": src_command["dataset"]["label_noise"],
                            "validation_split": src_command["dataset"][
                                "validation_split"
                            ],
                        },
                        "optimizer": src_command["optimizer"],
                        "precision": src_command["precision"],
                        "early_stopping": src_command["early_stopping"],
                    },
                }

                if get_value(old_experiment_table.butter):
                    dst_command["run"]["data"]["butter"] = {
                        k: True
                        for k, column in (
                            ("300_epoch_sweep", old_experiment_table.b_300_epoch_sweep),
                            ("30k_epoch_sweep", old_experiment_table.b_30k_epoch_sweep),
                            (
                                "learning_rate_sweep",
                                old_experiment_table.learning_rate_sweep,
                            ),
                            (
                                "label_noise_sweep",
                                old_experiment_table.label_noise_sweep,
                            ),
                            ("batch_size_sweep", old_experiment_table.batch_size_sweep),
                            (
                                "regularization_sweep",
                                old_experiment_table.regularization_sweep,
                            ),
                            ("optimizer_sweep", old_experiment_table.optimizer_sweep),
                            (
                                "learning_rate_batch_size_sweep",
                                old_experiment_table.learning_rate_batch_size_sweep,
                            ),
                            (
                                "size_adjusted_regularization_sweep",
                                old_experiment_table.size_adjusted_regularization_sweep,
                            ),
                        )
                        if get_value(column)
                    }

                experiment_id = json_to_uuid(dst_command["experiment"])
                if experiment_id not in experiment_ids:
                    experiment_ids[experiment_id] = marshal.demarshal(
                        dst_command["experiment"]
                    )

                result_records.append(
                    (
                        run_id,
                        get_value(job_status_table.start_time),
                        get_value(job_status_table.update_time),
                        get_value(job_status_table.worker),
                        get_value(job_status_table.error_count),
                        get_value(job_status_table.error),
                        get_value(job_status_table.parent),
                        Jsonb(dst_command),
                        experiment_id,
                        get_value(run_table.run_history),
                        get_value(run_table.run_extended_history),
                    )
                )

            except Exception as e:
                num_excepted += 1
                print(
                    f"failed to convert run {run_id}, experiment {get_value(experiment_table.experiment_id)} on Exception: {e}",
                    flush=True,
                )
                traceback.print_exc()

        input_columns = ColumnGroup(
            Column("id", "uuid"),
            Column("start_time", "timestamp with time zone"),
            Column("update_time", "timestamp with time zone"),
            Column("worker", "uuid"),
            Column("error_count", "smallint"),
            Column("error", "text"),
            Column("parent", "uuid"),
            Column("command", "jsonb"),
            Column("experiment_id", "uuid"),
            Column("history", "bytea"),
            Column("extended_history", "bytea"),
        )

        if len(result_records) > 0:
            migrate_query = SQL(
                """
WITH input_data AS (
    SELECT
        {input_cast}
    FROM
        ( VALUES {input_placeholders} ) AS input_data ({input_colums})
),
rsi AS (
    INSERT INTO run_status
        (queue, status, priority, id, start_time, update_time, worker, error_count, error, parent)
    SELECT
        -100, 2, 0, id, start_time, update_time, worker, error_count, error, parent
    FROM input_data
    ON CONFLICT (id) DO NOTHING
),
rdi AS (
    INSERT INTO run_data
        (id, command)
    SELECT
        id, command
    FROM input_data
    ON CONFLICT (id) DO NOTHING
),
hi AS (
    INSERT INTO {history}
        (id, experiment_id, history, extended_history)
    SELECT
        id, experiment_id, history, extended_history
    FROM input_data
    ON CONFLICT (id) DO NOTHING
)
SELECT 1;"""
            ).format(
                input_cast=input_columns.casting_sql,
                input_placeholders=sql_comma.join(
                    [SQL("({})").format(input_columns.placeholders)]
                    * len(result_records)
                ),
                input_colums=input_columns.columns_sql,
                history=schema.history.identifier,
            )

            with ConnectionManager(credentials) as connection:
                connection.execute("SET TIMEZONE to 'UTC';")
                connection.execute(
                    migrate_query,
                    list(chain(*result_records)),
                    binary=True,
                )

        num_converted = len(result_records)
        total_num_converted += num_converted
        total_num_excepted += num_excepted

        del result_records

        for experiment_id, experiment in experiment_ids.items():
            try:
                summary = experiment.summarize(
                    schema.get_experiment_run_histories(experiment_id)
                )
                if summary is not None:
                    schema.store_summary(experiment, experiment_id, summary)  # type: ignore
            except:
                print(f"Failed to summarize experiment {experiment_id}.")
                traceback.print_exc()

        print(
            f"Worker {worker_number} : {worker_id} committed {num_converted}, excepted {num_excepted} runs. Lifetime total: {total_num_converted} / {total_num_excepted}."
        )

        if num_converted <= 0 and num_excepted <= 0:
            break

    return total_num_converted


def main():
    # global schema, credentials

    parser = argparse.ArgumentParser()
    parser.add_argument("num_workers", type=int)
    parser.add_argument("block_size", type=int)
    args = parser.parse_args()

    num_workers = args.num_workers
    block_size = args.block_size

    pool = multiprocessing.ProcessPool(num_workers)
    results = pool.uimap(do_work, ((i, block_size) for i in range(num_workers)))
    total_num_converted = sum(results)
    print(f"Done. Converted {total_num_converted} runs.")
    pool.close()
    pool.join()
    print("Complete.")


if __name__ == "__main__":
    main()
