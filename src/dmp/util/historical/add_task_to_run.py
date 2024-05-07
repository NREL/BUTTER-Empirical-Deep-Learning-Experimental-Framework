from itertools import chain
import os
import jobqueue

from psycopg import ClientCursor
import pyarrow
from dmp.common import flatten
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup

from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.task.experiment.training_experiment import training_experiment_keys
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

from pprint import pprint
import uuid
from psycopg.sql import SQL, Literal, Identifier
from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder

from jobqueue.cursor_manager import CursorManager
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask
from dmp.layer.dense import Dense
from dmp.postgres_interface.postgres_compressed_result_logger import (
    PostgresCompressedResultLogger,
)

from dmp.logging.postgres_parameter_map_v1 import PostgresParameterMapV1
from dmp.model.dense_by_size import DenseBySize

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)

from dmp.marshaling import marshal

import pathos.multiprocessing as multiprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_workers", type=int)
    parser.add_argument("block_size", type=int)
    args = parser.parse_args()

    num_workers = int(args.num_workers)
    block_size = int(args.block_size)

    pool = multiprocessing.ProcessPool(num_workers)
    results = pool.uimap(do_work, ((i, block_size) for i in range(num_workers)))
    total_num_converted = sum(results)
    print(f"Done. Converted {total_num_converted} runs.")
    pool.close()
    pool.join()
    print("Complete.")


def do_work(args):
    worker_number, block_size = args

    credentials = jobqueue.load_credentials("dmp")
    schema = PostgresInterface(credentials)

    worker_id = str(worker_number) + str(uuid.uuid4())
    total_num_converted = 0
    total_num_excepted = 0
    print(f"Worker {worker_number} : {worker_id} started...")

    while True:  #  binary=True, scrollable=True
        num_converted = 0
        num_excepted = 0

        run = schema.run
        experiment = schema._experiment_table

        run_columns = ColumnGroup(
            run.run_id,
            run.experiment_id,
            run.slurm_job_id,
            run.task_version,
            run.num_nodes,
            run.num_cpus,
            run.num_gpus,
            run.gpu_memory,
            run.host_name,
            run.batch,
            run.run_data,
        )

        command_column = Column("command", "jsonb")
        experiment_attrs_json = Column("experiment_attrs_json", "jsonb")
        experiment_tags_json = Column("experiment_tags_json", "jsonb")
        select_columns = run_columns + ColumnGroup(
            command_column,
            experiment_attrs_json,
            experiment_tags_json,
        )

        update_columns = ColumnGroup(
            run.run_id,
            run.task,
        )

        get_and_lock_query = SQL(
            """
SELECT
    {run_columns},
    {job_data}.{command},
    (
        select
            jsonb_object_agg(
                kind,
                COALESCE(to_jsonb(value_bool), to_jsonb(value_int), to_jsonb(value_float), to_jsonb(value_str), value_json)
            ) experiment_data
        from attr
        where
            array[attr.attr_id] <@ {experiment}.experiment_attrs
    ) {experiment_attrs_json},
    (
		select
			jsonb_object_agg(
				kind,
				COALESCE(to_jsonb(value_bool), to_jsonb(value_int), to_jsonb(value_float), to_jsonb(value_str), value_json)
			   ) experiment_data
		from attr
		where
			array[attr.attr_id] <@ {experiment}.experiment_tags
	) {experiment_tags_json}
FROM
    (
        SELECT
            {run_columns}
        FROM
            {run}
        WHERE
            {task} IS NULL
            LIMIT {block_size}
            FOR UPDATE
            SKIP LOCKED
    ) {run},
    {job_data},
    {experiment}
WHERE
    {run}.{experiment_id} = {experiment}.{experiment_id}
	AND {job_data}.{id} = {run}.{run_id}
	AND jsonb_typeof({job_data}.{command}) = 'object'
;"""
        ).format(
            run_columns=run_columns.of(run.identifier),
            job_data=Identifier("job_data"),
            experiment=experiment.identifier,
            command=Identifier("command"),
            experiment_attrs_json=experiment_attrs_json.identifier,
            experiment_tags_json=experiment_tags_json.identifier,
            task=run.task.identifier,
            block_size=Literal(block_size),
            run=run.identifier,
            experiment_id=experiment.experiment_id.identifier,
            id=Identifier("id"),
            run_id=run.run_id.identifier,
        )

        with ConnectionManager(schema.credentials) as connection:
            with connection.transaction():
                run_updates = []
                with connection.cursor(binary=True) as cursor:
                    cursor.execute(get_and_lock_query, binary=True)
                    rows = list(cursor.fetchall())

                    run_updates = []
                    for row in rows:
                        task = row[select_columns[command_column]]
                        run_data = row[select_columns[run.run_data]]
                        experiment_attrs = row[select_columns[experiment_attrs_json]]
                        experiment_tags = row[select_columns[experiment_tags_json]]

                        print("task:")
                        pprint(task)
                        print("\nrun data")
                        pprint(run_data)
                        print("\nexperiment_attrs")
                        pprint(experiment_attrs)
                        print("\nexperiment_tags")
                        pprint(experiment_tags)

                        task["loss"] = experiment_attrs["loss"]

                        if not isinstance(task["model"]["output"], dict):
                            task["model"]["output"] = {}
                        task["model"]["output"] = task["model"]["inner"]
                        task["model"]["output"]["output_activation"] = experiment_attrs[
                            "model_output_activation"
                        ]
                        task["model"]["output"][
                            "kernel_initializer"
                        ] = experiment_attrs["model_output_kernel_initializer"]
                        task["model"]["output"]["units"] = experiment_attrs[
                            "model_output_units"
                        ]

                        if not isinstance(task["model"]["input"], dict):
                            task["model"]["input"] = {}
                        task["model"]["input"]["type"] = experiment_attrs["model_input"]
                        task["model"]["input"]["shape"] = experiment_attrs[
                            "model_input_shape"
                        ]

                        et = {}
                        if "experiment_tags" in task:
                            if isinstance(task["experiment_tags"], dict):
                                et.update(task["experiment_tags"])
                            del task["experiment_tags"]
                        et.update(experiment_tags)
                        task["tags"] = et

                        run_config = task["record"]
                        del task["record"]
                        task["run_config"] = run_config

                        run_data.update(
                            {
                                "task_version": row[select_columns[run.task_version]],
                                "slurm_job_id": row[select_columns[run.slurm_job_id]],
                                "num_nodes": row[select_columns[run.num_nodes]],
                                "num_cpus": row[select_columns[run.num_cpus]],
                                "num_gpus": row[select_columns[run.num_gpus]],
                                "gpu_memory": row[select_columns[run.gpu_memory]],
                                "host_name": row[select_columns[run.host_name]],
                                "ml_task": experiment_attrs["ml_task"],
                                "input_shape": experiment_attrs["input_shape"],
                                "output_shape": experiment_attrs["output_shape"],
                                "data_set_size": experiment_attrs["data_set_size"],
                                "test_set_size": experiment_attrs["test_set_size"],
                                "train_set_size": experiment_attrs["train_set_size"],
                                "num_free_parameters": experiment_attrs[
                                    "num_free_parameters"
                                ],
                                "validation_set_size": experiment_attrs[
                                    "validation_set_size"
                                ],
                            }
                        )

                        if "run_tags" in task and isinstance(task["run_tags"], dict):
                            run_data["tags"] = task["run_tags"]
                        else:
                            run_data["tags"] = None

                        task["runtime"] = run_data

                        run_updates.append(
                            (
                                row[select_columns[run.run_id]],
                                task,
                            )
                        )

                    if len(run_updates) > 0:
                        cursor.execute(
                            SQL(
                                """
UPDATE {run} SET
    {task} = {task}
FROM
    (VALUES {placeholders}) {values} ({update_columns})
WHERE
    {run}.{run_id} = {values}.{run_id}
                        ;"""
                            ).format(
                                run=run.identifier,
                                task=run.task.identifier,
                                values=Identifier("_values"),
                                placeholders=sql_comma.join(
                                    [SQL("({})").format(update_columns.placeholders)]
                                    * len(run_updates)
                                ),
                                update_columns=update_columns.columns_sql,
                                run_id=run.run_id.identifier,
                            ),
                            list(chain(*run_updates)),
                            binary=True,
                        )
                    num_converted = len(run_updates)
        total_num_converted += num_converted
        total_num_excepted += num_excepted
        print(
            f"Worker {worker_number} : {worker_id} committed {num_converted}, excepted {num_excepted} runs. Lifetime total: {total_num_converted} / {total_num_excepted}."
        )

        if num_converted <= 0 and num_excepted <= 0:
            break

    return total_num_converted


if __name__ == "__main__":
    main()
