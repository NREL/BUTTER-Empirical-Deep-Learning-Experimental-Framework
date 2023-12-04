import io
import os
from itertools import chain
from psycopg.types.json import Jsonb
from dmp.parquet_util import convert_bytes_to_dataframe, convert_dataframe_to_bytes

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
from dmp.uuid_tools import json_to_uuid


import pathos.multiprocessing as multiprocessing


dataset_path = "../summary_by_epoch/"
"""
drop table export_experiment;

CREATE TABLE export_experiment (
	experiment_id UUID NOT NULL PRIMARY KEY,
	order_number integer not null,
	status smallint NOT NULL DEFAULT 0,
	"dataset" text,
    "optimizer" text,
	"learning_rate" real,
    "batch_size" text,
    "l1" float,
	"l2" float,
	"label_noise" real,
    "shape" text,
    "depth" smallint,
	"size" integer
);


INSERT INTO export_experiment (
	order_number,
	experiment_id,
	dataset,
	optimizer,
	learning_rate,
	batch_size,
	l1,
	l2,
	label_noise,
	shape,
	depth,
	size
)
SELECT
	(ROW_NUMBER() OVER (ORDER BY
			   dataset,
			   optimizer,
			   learning_rate,
			   batch_size,
			   l1,
			   l2,
			   label_noise,
			   shape,
			   depth,
			   size)
	) order_number,
	e.*
FROM
(
SELECT
	experiment_id,
	(experiment #> '{dataset,name}')::text dataset,
	(experiment #> '{optimizer,class}')::text optimizer,
	(experiment #> '{optimizer,learning_rate}')::float learning_rate,
	(experiment #> '{fit,batch_size}')::smallint batch_size,
	(experiment #> '{dataset,label_noise}')::float label_noise,
	COALESCE((experiment #> '{model,inner,kernel_regularizer,l1}')::float, 0) l1,
	COALESCE((experiment #> '{model,inner,kernel_regularizer,l2}')::float, 0) l2,
	(experiment #> '{model,shape}')::text shape,
	(experiment #> '{model,depth}')::smallint depth,
	(experiment #> '{model,size}')::integer size
FROM experiment
WHERE TRUE
	AND experiment @> '{"data":{"butter":true}}'
ORDER BY
		   dataset,
		   optimizer,
		   learning_rate,
		   batch_size,
		   l1,
		   l2,
		   label_noise,
		   shape,
		   depth,
		   size
	) e;

create index on export_experiment (order_number) where status = 0;

UPDATE export_experiment SET
	status = 0
WHERE status <> 0;


CREATE TABLE IF NOT EXISTS public.export_experiment_block
(
    status smallint NOT NULL DEFAULT 0,
    dataset text COLLATE pg_catalog."default",
	shape text COLLATE pg_catalog."default",
    optimizer text COLLATE pg_catalog."default",
    learning_rate real,
    batch_size text COLLATE pg_catalog."default",
    regularizer text,
	has_label_noise boolean
)

alter table export_experiment add column regularizer text;
alter table export_experiment add column has_label_noise boolean;
update export_experiment set
	regularizer = (CASE
					WHEN l1>0 and l2 >0 THEN 'l1_l2'
					WHEN l1>0 and l2 <=0 THEN 'l1'
					WHEN l1<=0 and l2 >0 THEN 'l1'
					ELSE NULL
					END
				   ),
	has_label_noise = (label_noise > 0);


insert into c (
	dataset,
	shape,
	optimizer,
	learning_rate,
	batch_size,
	regularizer,
	has_label_noise)
select distinct
	dataset,
	shape,
	optimizer,
	learning_rate,
	batch_size,
	regularizer,
	has_label_noise
from export_experiment;

create index on export_experiment_block (status);

create index on export_experiment (dataset,
	shape,
	optimizer,
	learning_rate,
	batch_size,
	regularizer,
	has_label_noise, experiment_id);

    update export_experiment_block set status =0 where status <> 0;


"""


partition_cols = [
    "dataset",
    "shape",
    "optimizer",
    "learning_rate",
    "batch_size",
    "regularizer",
    "has_label_noise",
]


data_columns = [
    pyarrow.field("dataset", pyarrow.string(), nullable=False),
    pyarrow.field("depth", pyarrow.uint8(), nullable=False),
    pyarrow.field("epochs", pyarrow.uint32(), nullable=False),
    pyarrow.field("batch_size", pyarrow.uint32(), nullable=False),
    pyarrow.field("experiment_id", pyarrow.uint32(), nullable=False),
    # pyarrow.field("primary_sweep", pyarrow.bool_(), nullable=False),
    # pyarrow.field("300_epoch_sweep", pyarrow.bool_(), nullable=False),
    # pyarrow.field("30k_epoch_sweep", pyarrow.bool_(), nullable=False),
    # pyarrow.field("learning_rate_sweep", pyarrow.bool_(), nullable=False),
    # pyarrow.field("label_noise_sweep", pyarrow.bool_(), nullable=False),
    # pyarrow.field("batch_size_sweep", pyarrow.bool_(), nullable=False),
    # pyarrow.field("regularization_sweep", pyarrow.bool_(), nullable=False),
    # pyarrow.field('learning_rate_batch_size_sweep',
    #               pyarrow.bool_(), nullable=False),
    # pyarrow.field('size_adjusted_regularization_sweep',
    #               pyarrow.bool_(), nullable=False),
    # pyarrow.field("optimizer_sweep", pyarrow.bool_(), nullable=False),
    pyarrow.field("num_free_parameters", pyarrow.uint64(), nullable=False),
    pyarrow.field("widths", pyarrow.list_(pyarrow.uint32())),
    # pyarrow.field("network_structure", pyarrow.string()),
    pyarrow.field("num_runs", pyarrow.uint8(), nullable=False),
    pyarrow.field("label_noise", pyarrow.float32(), nullable=True),
    pyarrow.field("optimizer", pyarrow.string(), nullable=False),
    pyarrow.field("learning_rate", pyarrow.float32(), nullable=False),
    pyarrow.field("momentum", pyarrow.float32(), nullable=True),
    pyarrow.field("shape", pyarrow.string(), nullable=False),
    pyarrow.field("size", pyarrow.uint64(), nullable=False),
    pyarrow.field("task", pyarrow.string(), nullable=False),
    # pyarrow.field("kernel_regularizer", pyarrow.string(), nullable=True),
    pyarrow.field("l1", pyarrow.float32(), nullable=True),
    pyarrow.field("l2", pyarrow.float32(), nullable=True),
    pyarrow.field("epoch", pyarrow.list_(pyarrow.uint16()), nullable=False),
    pyarrow.field("regularizer", pyarrow.string(), nullable=False),
    pyarrow.field("has_label_noise", pyarrow.bool_(), nullable=False),
]

value_cols = [
    # "test_loss_best_count",
]

for key in [
    "test_loss",
    "test_accuracy",
    "test_mean_squared_error",
    "test_binary_crossentropy",
    "test_categorical_crossentropy",
]:
    for suffix1 in ["", "_best"]:
        for suffix2 in ["0", "25", "75", "100"]:
            value_cols.append(f"{key}{suffix1}_quantile_{suffix2}")

for prefix in ["train"]:
    for key in [
        "loss",
        "accuracy",
        "mean_squared_error",
        "binary_crossentropy",
        "categorical_crossentropy",
    ]:
        for suffix in ["_best_epoch_quantile_50", "_quantile_50"]:
            value_cols.append(f"{prefix}_{key}{suffix}")


schema_cols = data_columns + [
    pyarrow.field(name, pyarrow.list_(pyarrow.float32()), nullable=True)
    for name in value_cols
]

schema = pyarrow.schema(schema_cols)


def do_work(args):
    from dmp.marshaling import marshal

    # global schema, credentials

    worker_number, block_size = args

    credentials = load_credentials("dmp")
    database = PostgresInterface(credentials)

    worker_id = str(worker_number) + str(uuid.uuid4())
    total_num_converted = 0
    total_num_excepted = 0

    print(f"Worker {worker_number} : {worker_id} started...")

    experiment_filter = {"data": {"butter": True}}

    experiment_table = database._experiment_table
    butter_data_column = Column("butter_data", "jsonb")
    butter_e_data_column = Column("butter_e_data", "jsonb")
    selected_columns = ColumnGroup(
        experiment_table.experiment_id,
        experiment_table.old_experiment_id,
        experiment_table.num_runs,
        experiment_table.experiment,
        # butter_data_column,
        # butter_e_data_column,
        experiment_table.by_epoch,
        # experiment_table.by_loss,
    )

    query = SQL(
        """
WITH _selection AS (
SELECT
	experiment.experiment_id,
    experiment.old_experiment_id,
    experiment.num_runs,
    experiment.experiment,
    experiment.by_epoch,
    es.*
FROM
	(
		SELECT
            dataset,
            shape,
            optimizer,
            learning_rate,
            batch_size,
            regularizer,
            has_label_noise
		FROM export_experiment_block
		WHERE status = 0
		LIMIT {block_size}
		FOR UPDATE SKIP LOCKED
	) es
    INNER JOIN export_experiment eb ON (
            es.dataset = eb.dataset
            AND es.shape = eb.shape
            AND es.optimizer = eb.optimizer
            AND es.learning_rate = eb.learning_rate
            AND es.batch_size = eb.batch_size
            AND es.regularizer = eb.regularizer
            AND es.has_label_noise = eb.has_label_noise
    )
	INNER JOIN experiment ON (experiment.experiment_id = eb.experiment_id)
),
update_status AS (
UPDATE export_experiment_block eb SET
	status = 1
FROM _selection es
WHERE
	es.dataset = eb.dataset
    AND es.shape = eb.shape
    AND es.optimizer = eb.optimizer
    AND es.learning_rate = eb.learning_rate
    AND es.batch_size = eb.batch_size
    AND es.regularizer = eb.regularizer
    AND es.has_label_noise = eb.has_label_noise
)
SELECT {selected_columns} FROM _selection;
;"""
    ).format(
        selected_columns=selected_columns.columns_sql,
        # butter_data_path=Literal("{config,data,butter}"),
        # butter_e_data_path=Literal("{config,data,butter-e}"),
        experiment_filter=Literal(Jsonb(experiment_filter)),
        block_size=Literal(block_size),
    )

    while True:  #  binary=True, scrollable=True
        num_processed = 0
        num_excepted = 0

        source_records = []
        with ConnectionManager(credentials) as connection:
            with connection.cursor(binary=True) as cursor:
                cursor.execute(query, binary=True)
                source_records = list(cursor.fetchall())

        result_records = []
        status_updates = []

        for row in source_records:

            def get_value(column: Column):
                return row[selected_columns[column]]

            experiment_id = get_value(experiment_table.experiment_id)

            try:
                old_experiment_id = get_value(experiment_table.old_experiment_id)
                experiment = get_value(experiment_table.experiment)
                # butter_data = get_value(butter_data_column)
                # butter_e_data = get_value(butter_e_data_column)
                summary_df: pandas.DataFrame = convert_bytes_to_dataframe(
                    get_value(experiment_table.by_epoch)
                )  # type: ignore

                # print(experiment_id, old_experiment_id)
                # pprint(experiment)
                # pprint(butter_data)
                # pprint(butter_e_data)
                # print(summary_df)
                # pprint(summary_df.columns.to_list())

                del summary_df["test_loss_best_count"]

                # flatten per-epoch data into lists
                summary_df = summary_df.stack().reset_index(level=0, drop=True)  # type: ignore
                summary_df = (
                    summary_df.groupby(summary_df.index)
                    .apply(list)
                    .to_frame()
                    .transpose()
                )

                # populate per-run values
                summary_df["experiment_id"] = old_experiment_id
                summary_df["dataset"] = experiment["dataset"]["name"]
                summary_df["optimizer"] = experiment["optimizer"]["class"]
                summary_df["learning_rate"] = experiment["optimizer"]["learning_rate"]
                summary_df["batch_size"] = experiment["fit"]["batch_size"]

                l1 = 0.0
                try:
                    l1 = experiment["model"]["inner"]["kernel_regularizer"]["l1"]
                except Exception:
                    pass
                if l1 is None:
                    l1 = 0.0
                summary_df["l1"] = l1

                l2 = 0
                try:
                    l2 = experiment["model"]["inner"]["kernel_regularizer"]["l2"]
                except Exception:
                    pass
                if l2 is None:
                    l2 = 0.0
                summary_df["l2"] = l2

                if l1 > 0.0:
                    if l2 > 0.0:
                        summary_df["regularizer"] = "l1_l2"
                    else:
                        summary_df["regularizer"] = "l1"
                elif l2 > 0.0:
                    summary_df["regularizer"] = "l2"
                else:
                    summary_df["regularizer"] = None

                summary_df["label_noise"] = experiment["dataset"]["label_noise"]
                summary_df["has_label_noise"] = summary_df["label_noise"] > 0.0

                summary_df["shape"] = experiment["model"]["shape"]
                summary_df["depth"] = experiment["model"]["depth"]
                summary_df["size"] = experiment["model"]["size"]
                summary_df["task"] = experiment["data"]["ml_task"]

                summary_df["num_runs"] = get_value(experiment_table.num_runs)
                summary_df["num_free_parameters"] = experiment["data"][
                    "num_free_parameters"
                ]
                summary_df["widths"] = [
                    experiment["data"]["network_description"]["widths"]
                ]

                momentum = 0.0
                try:
                    momentum = experiment["optimizer"]["momentum"]
                except Exception:
                    pass
                summary_df["momentum"] = momentum

                # for key in [
                #     "primary_sweep",
                #     "300_epoch_sweep",
                #     "30k_epoch_sweep",
                #     "learning_rate_sweep",
                #     "label_noise_sweep",
                #     "batch_size_sweep",
                #     "regularization_sweep",
                #     "optimizer_sweep",
                # ]:
                #     summary_df[key] = False

                # if butter_data is not None:
                #     for key, value in butter_data.items():
                #         summary_df[key] = value

                # Thanks: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
                # def flatten(target):
                #     result = {}

                #     def do_flatten(prefix, target):
                #         if isinstance(target, dict):
                #             for key, value in target.items():
                #                 do_flatten(f"{prefix}_{key}", value)
                #         else:
                #             result[prefix] = target

                #     do_flatten("", target)
                #     return result

                # for key, value in flatten(butter_data).items():
                #     summary_df[key] = value

                # summary_df["experiment_id"] = old_experiment_id

                result_records.append(summary_df)
                status_updates.append((experiment_id, 2))
                num_processed += 1

            except Exception as e:
                num_excepted += 1
                status_updates.append((experiment_id, 3))
                print(
                    f"failed to convert {experiment_id} on Exception: {e}",
                    flush=True,
                )
                traceback.print_exc()

        input_columns = ColumnGroup(
            Column("experiment_id", "uuid"),
            Column("status", "smallint"),
        )

        if len(result_records) > 0:
            full_df = pandas.concat(result_records, ignore_index=True)

            if "" in full_df:
                del full_df[""]

            for column in schema_cols:
                if column.name not in full_df:
                    full_df[column.name] = None

            # pprint(full_df.columns.to_list())
            # print(full_df)
            # if len(full_df[(full_df["l1"] > 0.0) & (full_df["l2"] > 0.0)]) > 0:
            #     print(full_df)

            use_byte_stream_split = [name for name in value_cols]
            use_byte_stream_split_set = set(use_byte_stream_split)
            use_dictionary = [
                field.name
                for field in schema
                if field.name not in use_byte_stream_split_set
            ]

            column_encoding = {
                name: "BYTE_STREAM_SPLIT" for name in use_byte_stream_split
            }
            # column_encoding.update({name: "PLAIN_DICTIONARY" for name in use_dictionary})

            pyarrow.parquet.write_to_dataset(
                pyarrow.Table.from_pydict(
                    full_df,
                    schema=schema,
                ),
                root_path=dataset_path,
                schema=schema,
                partition_cols=partition_cols,
                # data_page_size=128 * 1024,
                compression="ZSTD",
                compression_level=20,
                use_dictionary=False,
                # use_byte_stream_split=use_byte_stream_split,
                # data_page_version='2.0',
                existing_data_behavior="overwrite_or_ignore",
                use_legacy_dataset=False,
                # write_batch_size=64,
                # dictionary_pagesize_limit=64*1024,
                column_encoding=column_encoding,
            )
            result_records = []

        #         if len(status_updates) > 0:
        #             migrate_query = SQL(
        #                 """
        # UPDATE export_experiment SET
        #     status = input_data.status
        # FROM
        #     (
        #         SELECT
        #             {input_cast}
        #         FROM
        #             ( VALUES {input_placeholders} ) AS input_data ({input_colums})) input_data
        # WHERE TRUE
        #     AND export_experiment.experiment_id = input_data.experiment_id
        # ;"""
        #             ).format(
        #                 input_colums=input_columns.columns_sql,
        #                 input_cast=input_columns.casting_sql,
        #                 input_placeholders=input_columns.placeholders_for_values(
        #                     len(status_updates)
        #                 ),
        #             )

        #             with ConnectionManager(credentials) as connection:
        #                 connection.execute(
        #                     migrate_query,
        #                     list(chain(*status_updates)),
        #                     binary=True,
        #                 )

        total_num_converted += num_processed
        total_num_excepted += num_excepted

        print(
            f"Worker {worker_number} : {worker_id} committed {num_processed}, excepted {num_excepted} runs. Lifetime total: {total_num_converted} / {total_num_excepted}."
        )

        if num_processed <= 0 and num_excepted <= 0:
            break

    return total_num_converted


def main():
    # global schema, credentials

    for k, v in {
        "display.max_rows": 9000,
        "display.min_rows": 40,
        "display.max_columns": None,
        "display.width": 300,
    }.items():
        pandas.set_option(k, v)

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
