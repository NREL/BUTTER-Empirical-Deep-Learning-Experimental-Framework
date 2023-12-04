import io
import os
from itertools import chain
from psycopg.types.json import Jsonb
from dmp.parquet_util import convert_dataframe_to_bytes

from dmp.task.experiment.training_experiment.training_experiment_keys import (
    TrainingExperimentKeys,
)

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
from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.task.experiment.experiment import Experiment
from dmp.uuid_tools import json_to_uuid

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)


import pathos.multiprocessing as multiprocessing

"""

"""


def do_work(args):
    from dmp.marshaling import marshal

    # global schema, credentials

    worker_number, block_size = args

    credentials = load_credentials("dmp")
    schema = PostgresInterface(credentials, 0)

    worker_id = str(worker_number) + str(uuid.uuid4())
    total_num_converted = 0
    total_num_excepted = 0
    print(f"Worker {worker_number} : {worker_id} started...")

    get_attsr_query = SQL(
        """
select
	attr_id, kind, COALESCE(to_jsonb(value_bool), to_jsonb(value_int), to_jsonb(value_float), to_jsonb(value_str), value_json) v
from attr
"""
    )

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

    attr_map = None

    while True:  #  binary=True, scrollable=True
        num_converted = 0
        num_excepted = 0

        source_records = []
        with ConnectionManager(credentials) as connection:
            connection.execute("SET TIMEZONE to 'UTC';")

            with connection.cursor(binary=True) as cursor:
                if attr_map is None:
                    cursor.execute(get_attsr_query)
                    attr_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

                cursor.execute(get_runs_query, binary=True)
                source_records = list(cursor.fetchall())

        result_records = []

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

                attrs = {}
                for attr_id in chain(
                    get_value(experiment_table.experiment_attrs),
                    get_value(experiment_table.experiment_tags),
                ):
                    kind, value = attr_map[attr_id]  # type: ignore
                    attrs[kind] = value

                # dataset_name = src_command["dataset"]["name"]
                # dsinfo = dataset_index[dataset_index["Dataset"] == dataset_name].iloc[0]
                # ml_task = MLTask.regression
                # num_outputs = 1

                # if dataset_name == "201_pol":
                #     ml_task = MLTask.classification
                #     num_outputs = 11
                # elif dataset_name == "294_satellite_image":
                #     ml_task = MLTask.classification
                #     num_outputs = 6
                # elif dsinfo["Task"] == "classification":
                #     ml_task = MLTask.classification
                #     num_outputs = int(dsinfo["n_classes"])
                #     if num_outputs == 2:
                #         num_outputs = 1

                # dataset_size = int(dsinfo["n_observations"])
                # test_split = float(src_command["dataset"]["test_split"])
                # train_set_size = math.floor(dataset_size * test_split)
                # test_set_size = dataset_size - train_set_size

                # shapes = dataset_shape_map[dataset_name]

                # input_shape = [
                #     shapes[0],
                # ]

                # output_activation = "softmax"
                # output_kernel_initializer = "GlorotUniform"
                # loss = "CategoricalCrossentropy"
                # loss_metric = "categorical_crossentropy"
                # if ml_task == MLTask.regression:
                #     output_activation = "sigmoid"
                #     output_kernel_initializer = "GlorotUniform"
                #     loss = "MeanSquaredError"
                #     loss_metric = "mean_squared_error"
                # elif ml_task == MLTask.classification:
                #     if num_outputs == 1:
                #         output_activation = "sigmoid"
                #         output_kernel_initializer = "GlorotUniform"
                #         loss = "BinaryCrossentropy"
                #         loss_metric = "binary_crossentropy"
                #     else:
                #         output_activation = "softmax"
                #         output_kernel_initializer = "GlorotUniform"
                #         loss = "CategoricalCrossentropy"
                #         loss_metric = "categorical_crossentropy"

                # history = parquet_to_dataframe(get_value(run_table.run_history))

                # for prefix in keys.measurement_prefixes:
                #     loss_key = prefix + "_" + keys.loss
                #     metric_key = prefix + "_" + loss_metric
                #     if (
                #         loss_key in history
                #         and metric_key in history
                #         and all(
                #             (x is y) or (x == y)
                #             for x, y in zip(history[loss_key], history[metric_key])
                #         )
                #     ):
                #         del history[metric_key]

                optimizer = {
                    "class": attrs["optimizer"],
                }

                if "optimizer_learning_rate" in attrs:
                    optimizer["learning_rate"] = attrs["optimizer_learning_rate"]

                if "optimizer_momentum" in attrs:
                    optimizer["momentum"] = attrs["optimizer_momentum"]

                if "optimizer_nesterov" in attrs:
                    optimizer["nesterov"] = attrs["optimizer_nesterov"]

                inner_kernel_regularizer = None
                if "model_inner_kernel_regularizer" in attrs:
                    inner_kernel_regularizer = {
                        "class": attrs["model_inner_kernel_regularizer"]
                    }
                    if "model_inner_kernel_regularizer_l1" in attrs:
                        inner_kernel_regularizer["l1"] = attrs[
                            "model_inner_kernel_regularizer_l1"
                        ]
                    if "model_inner_kernel_regularizer_l2" in attrs:
                        inner_kernel_regularizer["l2"] = attrs[
                            "model_inner_kernel_regularizer_l2"
                        ]

                dst_command = {
                    "run": {
                        "data": {
                            "batch": src_command["batch"],
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
                            "epochs": attrs["fit_epochs"],
                            "batch_size": attrs["fit_batch_size"],
                        },
                        "data": {
                            "ml_task": attrs["ml_task"],
                            "input_shape": attrs["input_shape"],
                            "output_shape": attrs["output_shape"],
                            "data_set_size": attrs["data_set_size"],
                            "test_set_size": attrs["test_set_size"],
                            "train_set_size": attrs["train_set_size"],
                            "network_description": {"widths": attrs["model_widths"]},
                            "num_free_parameters": get_value(
                                old_experiment_table.num_free_parameters
                            ),
                            "validation_set_size": attrs["validation_set_size"],
                        },
                        "loss": {"class": attrs["loss"]},
                        "type": "TrainingExperiment",
                        "model": {
                            "size": attrs["model_size"],
                            "type": attrs["model"],
                            "depth": attrs["model_depth"],
                            "inner": {
                                "type": attrs["model_inner"],
                                "units": -1,
                                "use_bias": attrs["model_inner_use_bias"],
                                "activation": attrs["model_inner_activation"],
                                "bias_constraint": attrs["model_inner_bias_constraint"],
                                "bias_initializer": attrs[
                                    "model_inner_bias_initializer"
                                ],
                                "bias_regularizer": attrs[
                                    "model_inner_bias_regularizer"
                                ],
                                "kernel_constraint": attrs[
                                    "model_inner_kernel_constraint"
                                ],
                                "kernel_initializer": attrs[
                                    "model_inner_kernel_initializer"
                                ],
                                "kernel_regularizer": inner_kernel_regularizer,
                                "activity_regularizer": attrs[
                                    "model_inner_activity_regularizer"
                                ],
                            },
                            "input": {
                                "name": "dmp_2",
                                "type": "Input",
                                "shape": attrs["input_shape"],
                            },
                            "shape": attrs["model_shape"],
                            "output": {
                                "type": attrs["model_output"],
                                "units": attrs["model_output_units"],
                                "use_bias": attrs["model_output_use_bias"],
                                "activation": attrs["model_output_activation"],
                                "bias_constraint": attrs[
                                    "model_output_bias_constraint"
                                ],
                                "bias_initializer": attrs[
                                    "model_output_bias_initializer"
                                ],
                                "bias_regularizer": attrs[
                                    "model_output_bias_regularizer"
                                ],
                                "kernel_constraint": attrs[
                                    "model_output_kernel_constraint"
                                ],
                                "kernel_initializer": {
                                    "class": attrs["model_output_kernel_initializer"]
                                },
                                "kernel_regularizer": inner_kernel_regularizer,
                                "activity_regularizer": attrs[
                                    "model_output_activity_regularizer"
                                ],
                            },
                            "search_method": attrs["model_search_method"],
                        },
                        "dataset": {
                            "name": attrs["dataset"],
                            "type": "DatasetSpec",
                            "method": attrs["dataset_method"],
                            "source": "pmlb",
                            "test_split": attrs["dataset_test_split"],
                            "label_noise": attrs["dataset_label_noise"],
                            "validation_split": attrs["dataset_validation_split"],
                        },
                        "optimizer": optimizer,
                        "precision": src_command["precision"],
                        "early_stopping": src_command["early_stopping"],
                    },
                }

                if get_value(old_experiment_table.butter):
                    dst_command["run"]["data"]["butter"] = {
                        k: True
                        for k, column in (
                            (
                                "primary_sweep",
                                old_experiment_table.primary_sweep,
                            ),
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
                    dst_command["experiment"]["data"]["butter"] = True
                elif "energy" in src_command["batch"].lower():
                    dst_command["experiment"]["data"]["butter-e"] = True
                else:
                    dst_command["experiment"]["data"]["historical"] = True

                experiment_id = json_to_uuid(dst_command["experiment"])
                # if experiment_id not in experiment_ids:
                #     experiment_ids[experiment_id] = marshal.demarshal(
                #         dst_command["experiment"]
                #     )

                result_records.append(
                    (
                        run_id,
                        # get_value(job_status_table.start_time),
                        # get_value(job_status_table.update_time),
                        # get_value(job_status_table.worker),
                        # get_value(job_status_table.error_count),
                        # get_value(job_status_table.error),
                        # get_value(job_status_table.parent),
                        Jsonb(dst_command),
                        experiment_id,
                        # convert_dataframe_to_bytes(history),
                        # get_value(run_table.run_extended_history),
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
            # Column("start_time", "timestamp with time zone"),
            # Column("update_time", "timestamp with time zone"),
            # Column("worker", "uuid"),
            # Column("error_count", "smallint"),
            # Column("error", "text"),
            # Column("parent", "uuid"),
            Column("command", "jsonb"),
            Column("experiment_id", "uuid"),
            # Column("history", "bytea"),
            # Column("extended_history", "bytea"),
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
rdi AS (
    UPDATE run_data SET
        command = input_data.command
    FROM input_data
    WHERE run_data.id = input_data.id
),
hi AS (
    UPDATE history SET
        experiment_id = input_data.experiment_id
    FROM input_data
    WHERE history.id = input_data.id
),
mu AS (
    UPDATE migration SET
        status = 2
    FROM input_data
    WHERE migration.run_id = input_data.id
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
            #             migrate_query = SQL(
            #                 """
            # WITH input_data AS (
            #     SELECT
            #         {input_cast}
            #     FROM
            #         ( VALUES {input_placeholders} ) AS input_data ({input_colums})
            # ),
            # rsi AS (
            #     INSERT INTO run_status
            #         (queue, status, priority, id, start_time, update_time, worker, error_count, error, parent)
            #     SELECT
            #         -100, 2, 0, id, start_time, update_time, worker, error_count, error, parent
            #     FROM input_data
            #     ON CONFLICT (id) DO NOTHING
            # ),
            # rdi AS (
            #     INSERT INTO run_data
            #         (id, command)
            #     SELECT
            #         id, command
            #     FROM input_data
            #     ON CONFLICT (id) DO NOTHING
            # ),
            # hi AS (
            #     INSERT INTO {history}
            #         (id, experiment_id, history, extended_history)
            #     SELECT
            #         id, experiment_id, history, extended_history
            #     FROM input_data
            #     ON CONFLICT (id) DO NOTHING
            # )
            # SELECT 1;"""
            #             ).format(
            #                 input_cast=input_columns.casting_sql,
            #                 input_placeholders=sql_comma.join(
            #                     [SQL("({})").format(input_columns.placeholders)]
            #                     * len(result_records)
            #                 ),
            #                 input_colums=input_columns.columns_sql,
            #                 history=schema.history.identifier,
            #             )

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

        # for experiment_id, experiment in experiment_ids.items():
        #     try:
        #         summary = experiment.summarize(
        #             schema.get_experiment_run_histories(experiment_id)
        #         )
        #         if summary is not None:
        #             schema.store_summary(experiment, experiment_id, summary)  # type: ignore
        #     except:
        #         print(f"Failed to summarize experiment {experiment_id}.")
        #         traceback.print_exc()

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
