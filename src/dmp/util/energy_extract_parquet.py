# %%
# %load_ext autoreload
# %autoreload 2
import numpy
import pandas
import io
import uuid
import psycopg.sql
import pyarrow
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.parquet 
import jobqueue
from jobqueue.connection_manager import ConnectionManager
import multiprocessing
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq


import json

import tqdm as tqmd

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

from typing import Callable, List

from psycopg import sql

import dmp.keras_interface.model_serialization as model_serialization
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.table import Table
from dmp.postgres_interface.element.column_group import ColumnGroup

from dmp.util.butter_e_export import *

pd.options.display.max_seq_items = None
credentials = jobqueue.load_credentials("dmp")

print(f"run vars {vars(run)}")

# %%
from psycopg import ClientCursor


print(f"run vars {vars(run)}")

columns = (
    run
    + ColumnGroup(*[c for c in job_status.columns if c.name != "id"])
    + job_data.command
)
print(columns.names)


def passthrough(row, index, value, column, data):
    data[column.name] = value


column_converters: List[Callable] = [passthrough for _ in columns]


def flatten_json(json_obj, destination=None, parent_key="", separator="_"):
    if isinstance(destination, dict):
        flattened = destination
    else:
        flattened = {}

    for key, value in json_obj.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key, separator=separator))
        else:
            flattened[new_key] = value
    return flattened


column_converters[
    columns.get_index_of(job_data.command)
] = lambda row, index, value, column, data: flatten_json(value, destination=data)
column_converters[
    columns.get_index_of(run.run_data)
] = lambda row, index, value, column, data: flatten_json(value, destination=data)


def parquet_to_dataframe(row, index, value, column, data):
    with io.BytesIO(value) as buffer:
        data[column.name] = (
            pyarrow.parquet.read_table(pyarrow.PythonFile(buffer, mode="r"))
            .to_pandas()
            .sort_values(by="epoch")
        )


column_converters[columns.get_index_of(run.run_history)] = parquet_to_dataframe
column_converters[columns.get_index_of(run.run_extended_history)] = parquet_to_dataframe


dfs = []


# %%
def get_json(**kwargs):
    # Given dictionary
    given_dict = {}
   
    # Update the given dictionary with user input
    given_dict.update(kwargs)

    # Convert the updated dictionary to JSON
    json_result = json.dumps(given_dict, indent=2)

    return json_result

# %%
def process_row(row, columns, column_converters):
    """
    Process a single row using the specified columns and column converters.

    Parameters:
    - row: The row to be processed.
    - columns: List of column names.
    - column_converters: List of functions to convert each column.

    Returns:
    - Processed DataFrame containing the row data.
    """
    row_data = {}
    for i, (column, column_converter) in enumerate(zip(columns, column_converters)):
        column_converter(row, i, row[i], column, row_data)

    row_df = row_data["run_history"]
    row_df = row_df.join(
        row_data["run_extended_history"], on="epoch", how="left", rsuffix="_"
    )

    for k in ("run_history", "run_extended_history"):
        del row_data[k]

    for k, v in row_data.items():
        if k in row_df:
            pass
        if isinstance(v, list):
            row_df[k] = [v] * len(row_df)
        else:
            row_df[k] = v

    for index, row in row_df.iterrows():
        for column in row_df.columns:
            if isinstance(row[column], uuid.UUID):
                row_df.at[index, column] = str(row[column])
                
    return row_df

# %%
def get_data(query_search):
    with ConnectionManager(credentials) as connection:
        query = psycopg.sql.SQL(
            """
    SELECT
        {columns}
    FROM
        {run},
        {job_status},
        {job_data}
    WHERE
        {run}.batch LIKE {pattern}
        AND {job_status}.id = {run}.run_id
        AND {job_status}.id = {job_data}.id
        AND {job_status}.status = 2
        AND {job_data}.command @> {json_data}::jsonb
    ORDER BY
        experiment_id, run_id;
    """
        ).format(
            columns=columns.columns_sql,
            run=run.identifier,
            job_status=job_status.identifier,
            job_data=job_data.identifier,
            pattern=sql.Literal("%energy%"),
            json_data=sql.Literal(query_search),
        )

        with ClientCursor(connection) as c:
            c.mogrify(query)

        with connection.cursor(binary=True) as cursor:
            cursor.execute(query, binary=True)

            dfs = []
            for row in tqdm.tqdm(cursor, total=cursor.rowcount, desc="Loading data"):
                processed_row_df = process_row(row, columns, column_converters)
                dfs.append(processed_row_df)

    data = pandas.concat(dfs)
    
    return data

# %%
def save_data(queries:list):
    for query in tqmd.tqdm(queries):
        data = get_data(query)
        # convert to pyarrow table
        table = pa.Table.from_pandas(data)

        # write to distributed parquet file saved as ['name','depth','size','shape']
        pq.write_to_dataset(table, root_path='/projects/gcomp/jgafur/dataset/', partition_cols=['name','shape','depth'])
        del data

def save_parallel(query):
    data = get_data(query)
    table = pa.Table.from_pandas(data)
    pq.write_to_dataset(table, root_path='/projects/gcomp/jgafur/dataset/', partition_cols=['name', 'shape', 'depth'])
    del data

def save_data_parallelized(queries):
    with Pool() as pool:
        list(tqmd.tqdm(pool.imap(save_parallel, queries), total=len(queries)))

# %%
# query_search1 = get_json( dataset={"name": "sleep"},
#         )

query_search2 = get_json(dataset={"name":"banana"},
        )

# query_search3 = get_json(dataset={"name":"connect_4"},
#         )

query_search4 = get_json(dataset={"name":"mnist"},
        )

query_search5 = get_json(dataset={"name":"nursery"},
        )

# query_search6 = get_json(dataset={"name":"splice"},
#         )

# query_search7 = get_json(dataset={"name":"wine_quality_white"},
#         )

query_search8 = get_json(dataset={"name":"201_pol"},
        )

# query_search9 = get_json(dataset={"name":"294_satellite_image"},
#         )

# query_search10 = get_json(dataset={"name":"505_tecator"},
#         )

# query_search11 = get_json(dataset={"name":"529_pollen"},                 
#         )

# query_search12 = get_json(dataset={"name":"537_houses"},
#         )

query_search13 = get_json(dataset={"name":"adult"},
        )     


save_data_parallelized([query_search2, query_search4, query_search5, query_search8, query_search13])
# %%
# save_data_parallelized([query_search1])

# %%



