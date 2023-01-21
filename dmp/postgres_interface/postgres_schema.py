from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union

import itertools
import io
import uuid
import hashlib

# import psycopg
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb, Json, set_json_dumps
import pyarrow
import pyarrow.parquet
import simplejson

from dmp.parquet_util import make_pyarrow_schema
from dmp.postgres_interface.attr import Attr
from dmp.postgres_interface.attribute_value_type import AttributeValueType
from dmp.postgres_interface.postgres_attr_map import PostgresAttrMap


def json_dump_function(value: Any) -> str:
    return simplejson.dumps(value, sort_keys=True, separators=(',', ':'))


set_json_dumps(json_dump_function)

comma_sql = SQL(',')  # sql comma delimiter
placeholder_sql = SQL('%s')  # sql placeholders / references

class ColumnGroup():
    _columns: Sequence[str]
    _types: Sequence[str]
    _index: Dict[str, int]

    def __init__(self, columns_and_types: Iterable[Tuple[str, str]]) -> None:
        self._columns = tuple((name for name, type in columns_and_types))
        self._types = tuple((type for name, type in columns_and_types))
        self._index = {name: i for i, name in enumerate(self._columns)}

    def __getitem__(self, key: Union[str, int]) -> Union[str, int]:
        if isinstance(key, str):
            return self._index[key]
        return self._columns[key]
            

    @staticmethod
    def concatenate(groups: Iterable['ColumnGroup']) -> 'ColumnGroup':
        return ColumnGroup(
            tuple(
                itertools.chain(*(group.columns_and_types
                                  for group in groups))))

    def __add__(self, other: 'ColumnGroup') -> 'ColumnGroup':
        return self.concatenate((self, other))

   
    @property
    def columns(self) -> Sequence[str]:
        return self._columns

    @property
    def types(self) -> Sequence[str]:
        return self._types

    @property
    def columns_and_types(self) -> Iterable[Tuple[str, str]]:
        return zip(self._columns, self._types)

    @property
    def column_identifiers(self)->Sequence[Identifier]:
        return tuple((Identifier(name) for name in self._columns))

    @property
    def columns_sql(self) -> Composed:
        return comma_sql.join(self.column_identifiers)

    @property
    def casting_sql(self) -> Composed:
        return comma_sql.join((
            SQL('{}::{}').format(Identifier(name), SQL(type))  # type: ignore
            for name, type in self.columns_and_types
        ))

    @property
    def placeholders(self) -> Composed:
        return comma_sql.join([placeholder_sql] * len(self._columns))

    def columns_from(self, table_name:Identifier)->Composed:
        return comma_sql.join((
            SQL('{}.{}').format(table_name, column)  # type: ignore
            for column in self.column_identifiers
        ))

    def extract_column_values(
        self,
        source: Dict[str, Any],
    ) -> List[Any]:
        result = []
        for name, type_name in self.columns_and_types:
            value = source.pop(name, None)
            if type_name == 'jsonb' and value is not None:
                value = Jsonb(value)
            elif type_name == 'json' and value is not None:
                value = Json(value)
            else:
                pass
            result.append(value)
        return result

    



class TableData():
    _name: str
    _groups: Dict[str, ColumnGroup]

    def __init__(
        self,
        name: str,
        column_groups: Dict[str, ColumnGroup],
    ) -> None:
        self._name = name
        self._groups = column_groups

    @property
    def name(self) -> str:
        return self._name

    @property
    def name_sql(self) -> Identifier:
        return Identifier(self._name)

    def __getitem__(self, group_name: str) -> ColumnGroup:
        return self._groups[group_name]


class PostgresSchema:
    credentials: Dict[str, Any]

    attr: TableData
    experiment: TableData
    run: TableData

    log_result_record_query: Composed
    log_query_suffix: Composed
    attribute_map: 'PostgresAttrMap'

    def __init__(
        self,
        credentials: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.credentials = credentials

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
                'id': ColumnGroup([
                    ('experiment_uid', 'uuid'),
                ]),
                'attr': ColumnGroup([
                    ('experiment_attrs', 'integer[]'),
                ]),
                'value': ColumnGroup([
                    ('experiment_id', 'integer'),
                ]),
            })

        self.run = Identifier(
            'run2',
            {
                'experiment':
                ColumnGroup([('experiment_uid', 'uuid')]),
                'time':
                ColumnGroup([('run_timestamp', 'timestamp')]),
                'value':
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

        self.experiment_summary_progress = TableData(
            'experiment_summary_progress',
            [],
        )

        # initialize parameter map
        from dmp.postgres_interface.postgres_schema import PostgresAttrMap
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


from dmp.postgres_interface.postgres_schema import PostgresAttrMap