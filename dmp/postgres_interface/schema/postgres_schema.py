from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union

import io
import uuid
import hashlib
import numpy
import pandas

# import psycopg
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb, Json
import pyarrow
import pyarrow.parquet
from dmp import parquet_util

from dmp.parquet_util import make_pyarrow_table_from_dataframe
from dmp.postgres_interface.attribute_value_type import AttributeValueType

from dmp.postgres_interface.postgres_interface_common import json_dump_function
from dmp.postgres_interface.schema.attr_table import AttrTable
from dmp.postgres_interface.schema.experiment_summary_table import ExperimentSummaryTable
from dmp.postgres_interface.schema.experiment_table import ExperimentTable
from dmp.postgres_interface.schema.run_table import RunTable


class PostgresSchema:
    credentials: Dict[str, Any]

    experiment_id_column: str

    run: RunTable
    attr: AttrTable
    experiment: ExperimentTable
    experiment_summary: ExperimentSummaryTable

    log_query_suffix: Composed
    attribute_map: 'PostgresAttrMap'

    def __init__(
        self,
        credentials: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.credentials = credentials

        self.run = RunTable()
        self.attr = AttrTable()
        self.experiment = ExperimentTable()
        self.experiment_summary = ExperimentSummaryTable()

        # initialize parameter map
        from dmp.postgres_interface.postgres_attr_map import PostgresAttrMap
        self.attribute_map = PostgresAttrMap(self)

    def str_to_uuid(self, target: str) -> uuid.UUID:
        return uuid.UUID(hashlib.md5(target.encode('utf-8')).hexdigest())

    def json_to_uuid(self, target: Any) -> uuid.UUID:
        return self.str_to_uuid(json_dump_function(target))

    def make_experiment_id(self, experiment_attrs: Iterable[int]) -> uuid.UUID:
        return self.str_to_uuid('{' +
                                ','.join(str(i)
                                         for i in experiment_attrs) + '}')

    def convert_dataframe_to_bytes(
        self,
        dataframe: Optional[pandas.DataFrame],
    ) -> Optional[bytes]:
        if dataframe is None:
            return None

        # for column in dataframe.columns:
        #     print(f'col: {column} type: {dataframe[column].dtype} nptype: {dataframe[column].to_numpy().dtype} ')

        # Must do this to avoid a bug in pyarrow reading nulls in
        # byte stream split columns.
        # see https://github.com/apache/arrow/issues/28737
        # and https://issues.apache.org/jira/browse/ARROW-13024
        # for c in use_byte_stream_split:
        #     dataframe[c].fillna(value=numpy.nan, inplace=True)

        print(f'convert_dataframe_to_bytes')
        print(dataframe)
        print([dataframe[c].to_numpy().dtype for c in dataframe.columns])

        table, use_byte_stream_split = make_pyarrow_table_from_dataframe(dataframe)

        data = None
        with io.BytesIO() as buffer:
            parquet_util.write_parquet_table(
                table,
                buffer,
                use_byte_stream_split,
            )
            data = buffer.getvalue()

        # try:
        #     df = self.convert_bytes_to_dataframe(data)
        # except Exception as e:
        #     print(table, flush=True)
        #     raise e
        return data

    def convert_bytes_to_dataframe(
        self,
        data: Optional[bytes],
    ) -> Optional[pandas.DataFrame]:
        if data is None:
            return None

        with io.BytesIO(data) as b:
            pyarrow_file = pyarrow.PythonFile(b, mode='r')
            parquet_table = pyarrow.parquet.read_table(pyarrow_file, )
            return parquet_table.to_pandas()


from dmp.postgres_interface.postgres_attr_map import PostgresAttrMap