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

from dmp.parquet_util import make_pyarrow_schema_from_panads
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

    def make_experiment_id(self,
                            experiment_attrs: Iterable[int]) -> uuid.UUID:
        return self.str_to_uuid('{' +
                                ','.join(str(i)
                                         for i in experiment_attrs) + '}')

    def convert_dataframe_to_bytes(
        self,
        dataframe: Optional[pandas.DataFrame],
    ) -> Optional[bytes]:
        if dataframe is None:
            return None
        dataframe = dataframe.reset_index()
        schema, use_byte_stream_split = make_pyarrow_schema_from_panads(
            dataframe)

        # Must do this to avoid a bug in pyarrow reading nulls in
        # byte stream split columns.
        # see https://github.com/apache/arrow/issues/28737
        # and https://issues.apache.org/jira/browse/ARROW-13024
        for c in use_byte_stream_split:
            dataframe[c].fillna(value=numpy.nan, inplace=True)

        table = pyarrow.Table.from_pandas(
            dataframe,
            schema=schema,
            preserve_index=False,
        )

        with io.BytesIO() as buffer:
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
            return buffer.getvalue()

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