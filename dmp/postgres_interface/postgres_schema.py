from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union

import io
import uuid
import hashlib

# import psycopg
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb, Json
import pyarrow
import pyarrow.parquet

from dmp.parquet_util import make_pyarrow_schema
from dmp.postgres_interface.attribute_value_type import AttributeValueType

from dmp.postgres_interface.column_group import ColumnGroup
from dmp.postgres_interface.postgres_interface_common import json_dump_function
from dmp.postgres_interface.table_data import TableData


class PostgresSchema:
    credentials: Dict[str, Any]

    experiment_uid_column: str

    attr: TableData
    experiment: TableData
    run: TableData
    experiment_summary: TableData

    log_result_record_query: Composed
    log_query_suffix: Composed
    attribute_map: 'PostgresAttrMap'

    def __init__(
        self,
        credentials: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.credentials = credentials

        self.experiment_uid_column = 'experiment_uid'
        self.experiment_uid_group = ColumnGroup([(self.experiment_uid_column,
                                                  'uuid')])

        self.attr = TableData(
            'attr', {
                'id':
                ColumnGroup([
                    ('attr_id', 'integer'),
                ]),
                'index':
                ColumnGroup([
                    ('value_type', 'smallint'),
                    ('kind', 'text'),
                ]),
                'value':
                ColumnGroup([(
                    attribute_type.sql_column,
                    attribute_type.sql_type,
                ) for attribute_type in AttributeValueType
                             if attribute_type.sql_column is not None]),
                'json':
                ColumnGroup([
                    ('value_json', 'jsonb'),
                ]),
            })

        self.experiment = TableData(
            'experiment2', {
                'uid': self.experiment_uid_group,
                'attrs': ColumnGroup([
                    ('experiment_attrs', 'integer[]'),
                ]),
                'values': ColumnGroup([
                    ('experiment_id', 'integer'),
                ]),
            })

        self.run = TableData(
            'run2',
            {
                self.experiment_uid_column:
                self.experiment_uid_group,
                'timestamp':
                ColumnGroup([('run_timestamp', 'timestamp')]),
                'values':
                ColumnGroup([
                    ('run_id', 'uuid'),
                    ('job_id', 'uuid'),
                    ('seed', 'bigint'),
                    ('slurm_job_id', 'bigint'),
                    ('task_version', 'smallint'),
                    ('num_nodes', 'smallint'),
                    ('num_cpus', 'smallint'),
                    ('num_gpus', 'smallint'),
                    ('gpu_memory', 'integer'),
                    ('host_name', 'text'),
                    ('batch', 'text'),
                ]),  # type: ignore
                'data':
                ColumnGroup([('run_data', 'jsonb')]),
                'history':
                ColumnGroup([('run_history', 'bytea')]),
            })

        self.experiment_summary = TableData(
            'experiment_summary',
            {
                self.experiment_uid_column:
                self.experiment_uid_group,
                'last_run_timestamp':
                ColumnGroup([
                    ('last_run_timestamp', 'timestamp'),
                ]),
                'run_update_limit':
                ColumnGroup([
                    ('run_update_limit', 'timestamp'),
                ]),
                'data':
                ColumnGroup([
                    ('core_data', 'bytea'),
                    ('extended_data', 'bytea'),
                ])
            },
        )

        # initialize parameter map
        from dmp.postgres_interface.postgres_attr_map import PostgresAttrMap
        self.attribute_map = PostgresAttrMap(self)

    def str_to_uuid(self, target: str) -> uuid.UUID:
        return uuid.UUID(hashlib.md5(target.encode('utf-8')).hexdigest())

    def json_to_uuid(self, target: Any) -> uuid.UUID:
        return self.str_to_uuid(json_dump_function(target))

    def make_experiment_uid(self,
                            experiment_attrs: Iterable[int]) -> uuid.UUID:
        return self.str_to_uuid('{' +
                                ','.join(str(i)
                                         for i in experiment_attrs) + '}')

    def make_history_bytes(
        self,
        history: dict,
        buffer: io.BytesIO,
    ) -> None:
        schema, use_byte_stream_split = make_pyarrow_schema(history.items())

        table = pyarrow.Table.from_pydict(history, schema=schema)

        pyarrow_file = pyarrow.PythonFile(buffer)

        pyarrow.parquet.write_table(
            table,
            pyarrow_file,
            data_page_size=8 * 1024,
            compression='ZSTD',
            compression_level=12,
            use_dictionary=False,
            use_byte_stream_split=use_byte_stream_split,  # type: ignore
            version='2.6',
            data_page_version='2.0',
            write_statistics=False,
        )


from dmp.postgres_interface.postgres_attr_map import PostgresAttrMap