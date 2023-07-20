import io
from typing import Dict, Iterable, Optional, Sequence, Tuple, Type, Union, List
import pandas
import pyarrow
from numpy import issubdtype, ndarray
import numpy
import pandas.core.indexes.range


def make_pyarrow_table_from_dataframe(
    dataframe: pandas.DataFrame,
) -> Tuple[pyarrow.Table, List[str]]:
    if not isinstance(dataframe.index, pandas.core.indexes.range.RangeIndex):
        dataframe = dataframe.reset_index()

    columns = list(dataframe.columns)

    def to_numpy(column):
        array = dataframe[column].to_numpy()
        dtype = array.dtype
        if (
            numpy.issubdtype(dtype, numpy.floating)
            and dtype != numpy.float32
            and dtype != numpy.float16
        ):
            array = array.astype(numpy.float32)
        elif dtype == object:
            has_bool = False
            has_int = False
            has_float = False
            has_str = False
            has_null = False

            for v in array:
                value_type = type(v)
                has_bool |= isinstance(value_type, bool)
                has_int |= isinstance(value_type, int)
                has_float |= isinstance(value_type, float)
                has_str |= isinstance(value_type, str)
                has_null |= v is None

            if has_null:
                if has_str:
                    array = array.astype("U")
                elif has_float:
                    array = array.astype(numpy.float32)
                elif has_int:
                    print(f'NULLABLE INT ----------------------------')
                    array = array.astype(numpy.obj)
                elif has_bool:
                    print(f'NULLABLE Bool ----------------------------')
                    array = array.astype(numpy.B)

        return array

    return make_pyarrow_table_from_numpy(
        [str(column) for column in columns],
        [to_numpy(column) for column in columns],
    )


def convert_dataframe_to_bytes(
    dataframe: Optional[pandas.DataFrame],
) -> Optional[bytes]:

    print(f'convert_dataframe_to_bytes\n{dataframe}')
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

    # print(f'convert_dataframe_to_bytes')
    # print(dataframe)
    # print([dataframe[c].to_numpy().dtype for c in dataframe.columns])

    table, use_byte_stream_split = make_pyarrow_table_from_dataframe(dataframe)

    print(f'convert_dataframe_to_bytes : pyarrow:\n{table}')

    data = None
    with io.BytesIO() as buffer:
        write_parquet_table(
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
    data: Optional[bytes],
) -> Optional[pandas.DataFrame]:
    if data is None:
        return None

    import pyarrow.types as types
    with io.BytesIO(data) as buffer:
        pqt = read_parquet_table(buffer)

        cols = {}
        schema = pqt.schema
        for name in schema.names:
            field = schema.field(name)
            _type = field.type
            column = pqt.column(name)

            pandas_dtype = None
            if field.nullable:
                if types.is_integer(_type):
                    pandas_dtype = pandas.Int64Dtype
                    if types.is_int8(_type):
                        pandas_dtype = pandas.Int8Dtype
                    if types.is_int16(_type):
                        pandas_dtype = pandas.Int8Dtype
                    if types.is_int32(_type):
                        pandas_dtype = pandas.Int8Dtype
                elif types.is_boolean(_type):
                    pandas_dtype = pandas.BooleanDtype

                if pandas_dtype is not None:
                    cols[name] = pandas.Series(pqt.column(name).to_pylist(), dtype=pandas_dtype)
                else:
                    cols[name] = pandas.Series(pqt.column(name).to_pylist())
            else:
                cols[name] = column.to_numpy()

        df = pandas.DataFrame(cols)
        print(f'-------------------------------------------------------------------\nconvert_bytes_to_dataframe:\n{pqt}\npandas:\n{df}')
        return df

        # def type_mapper(pq_dtype):
        #     if types.is_integer(pq_dtype):
        #         return pandas.Int64Dtype()
        #     return None
        # return read_parquet_table(buffer).to_pandas(
        #     types_mapper=type_mapper,
        #     integer_object_nulls=True,
        # )


def make_dataframe_from_dict(data: Dict[str, Iterable]) -> pandas.DataFrame:
    cols = {}
    for name, col in data.items():
        cols[name] = numpy.array(col)
    return pandas.DataFrame(cols)


def make_pyarrow_table_from_numpy(
    columns: Sequence[str],
    numpy_arrays: Sequence[numpy.ndarray],
    nan_to_none: bool = True,
) -> Tuple[pyarrow.Table, List[str]]:
    schema, use_byte_stream_split, column_data = make_pyarrow_schema_from_dict(
        zip(columns, numpy_arrays),
        nan_to_none=nan_to_none,
    )  # type: ignore
    return (
        pyarrow.Table.from_arrays([data for name, data in column_data], schema=schema),
        use_byte_stream_split,
    )


def make_pyarrow_schema_from_dict(
    columns: Iterable[Tuple[str, Union[list, ndarray]]],
    nan_to_none: bool = True,
) -> Tuple[pyarrow.Schema, List[str], List[Tuple[str, Union[list, ndarray]]]]:
    fields = []
    use_byte_stream_split = []

    result_columns = []
    for name, values in columns:
        (
            pyarrow_type,
            nullable,
            use_byte_stream_split_,
            values,
        ) = get_pyarrow_type_mapping(
            values,
            nan_to_none=nan_to_none,
        )
        result_columns.append((name, values))

        if pyarrow_type is None:
            continue

        if use_byte_stream_split_:
            use_byte_stream_split.append(name)

        fields.append(pyarrow.field(name, pyarrow_type, nullable=nullable))

    return pyarrow.schema(fields), use_byte_stream_split, result_columns


def _check_type(t, x):
    return isinstance(t, Type) and issubclass(t, x)


def get_pyarrow_type_mapping(
    values: Union[list, ndarray],
    nan_to_none: bool = True,
) -> Tuple[pyarrow.DataType, bool, bool, Union[list, ndarray]]:

    def is_null(v):
        if v is None:
            return True
        if isinstance(v, float):
            return nan_to_none and numpy.isnan(v)

        return pandas.isna(v)

    nullable = False
    nullable = any(
        is_null(v)
        for v in values
    )

    use_byte_stream_split = False

    t = None
    if isinstance(values, ndarray):
        t = values.dtype
    else:
        if len(values) == 0:
            return None, nullable, use_byte_stream_split, values
        for v in values:
            if v is not None:
                t = type(v)

    def check_integer():
        nonlocal dst_type, nullable
        hi = max(filter(lambda v: not is_null(v), values))
        lo = min(filter(lambda v: not is_null(v), values))
        if hi < (2**7 - 1) and lo > (-(2**7)):
            dst_type = pyarrow.int8()
        elif hi < (2**15 - 1) and lo > (-(2**15)):
            dst_type = pyarrow.int16()
        elif hi < (2**31 - 1) and lo > (-(2**31)):
            dst_type = pyarrow.int32()
        else:
            dst_type = pyarrow.int64()

    dst_type = t
    if _check_type(t, bool) or numpy.issubdtype(t, bool):
        dst_type = pyarrow.bool_()
    elif _check_type(t, int) or numpy.issubdtype(t, numpy.integer):
        check_integer()
    elif _check_type(t, float) or numpy.issubdtype(t, numpy.floating):
        dst_type = pyarrow.float32()
        if nan_to_none and numpy.any(numpy.isnan(values)):
            values = [None if numpy.isnan(v) else v for v in values]
            nullable = True
        use_byte_stream_split = True
        # print(f'check float {dst_type} {nullable} {use_byte_stream_split} {nan_to_none}')
    elif (
        _check_type(t, str)
        or numpy.issubdtype(t, numpy.string_)
        or numpy.issubdtype(t, numpy.str_)
    ):
        dst_type = pyarrow.string()
    elif _check_type(t, list):
        element_type = next(
            (
                et
                for et in (get_pyarrow_type_mapping(v) for v in values)
                if et is not None
            ),
            None,
        )
        if element_type is None:
            return None, nullable, use_byte_stream_split, values
        else:
            # dst_type = pyarrow.list_(get_pyarrow_type_mapping(element_type[0]))
            raise NotImplemented()
    else:
        false_types = {
            "bool": lambda v: isinstance(v, bool),
            "int": lambda v: isinstance(v, int),
            "float": lambda v: isinstance(v, float),
            "nan": lambda v: isinstance(v, float) and numpy.isnan(v),
            "str": lambda v: isinstance(v, str),
            "none": lambda v: v is None,
        }
        true_types = set()

        for v in values:
            to_remove = None
            for k, f in false_types.items():
                if f(v):
                    true_types.add(k)
                    if to_remove is None:
                        to_remove = [k]
                    else:
                        to_remove.append(k)

            if to_remove is not None:
                for k in to_remove:
                    false_types.pop(k)

            if len(false_types) == 0:
                break

        nullable = (
            nullable or ("none" in true_types) or ("nan" in true_types and nan_to_none)
        )

        if "str" in true_types:
            dst_type = pyarrow.string()
        elif "float" in true_types:
            dst_type = pyarrow.float32()
            use_byte_stream_split = True
            if nan_to_none and nullable:
                values = [None if v is None else v for v in values]
        elif "int" in true_types:
            check_integer()
        elif "bool" in true_types:
            dst_type = pyarrow.bool_()
        else:
            raise NotImplementedError(f"Unhandled type {t}.")

    return dst_type, nullable, use_byte_stream_split, values


def truncate_least_significant_bits(
    source: ndarray,
    bits_to_trim: int,
) -> ndarray:
    int_type = numpy.int64
    source_type = source.dtype
    if source_type == numpy.float32:
        int_type = numpy.int32
    elif source_type == numpy.float64:
        int_type = numpy.int64
    else:
        raise NotImplementedError(f"Unsupported dtype {source_type}.")

    significand, exponent = numpy.frexp(source)
    significand = numpy.bitwise_and(
        significand.view(int_type),
        numpy.bitwise_not(numpy.array([int(2 << bits_to_trim) - 1], dtype=int_type)),
    ).view(source_type)
    return numpy.ldexp(significand, exponent)


def write_parquet_table(
    table,
    file,
    use_byte_stream_split,
    use_dictionary=False,
    # root_path=dataset_path,
    # schema=schema,
    # partition_cols=partition_cols,
    data_page_size=8 * 1024,
    # compression='BROTLI',
    # compression_level=8,
    compression="ZSTD",
    # compression_level=12,
    compression_level=12,
    version="2.6",
    data_page_version="2.0",
    # existing_data_behavior='overwrite_or_ignore',
    # use_legacy_dataset=False,
    write_statistics=False,
    # write_batch_size=64,
    # dictionary_pagesize_limit=64*1024,
    **kwargs,
):
    pyarrow.parquet.write_table(
        table,
        pyarrow.PythonFile(file),
        use_byte_stream_split=use_byte_stream_split,
        data_page_size=data_page_size,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=use_dictionary,
        version=version,
        data_page_version=data_page_version,
        write_statistics=write_statistics,
        store_schema=True,
        **kwargs,
    )


def read_parquet_table(file) -> pyarrow.Table:
    pyarrow_file = pyarrow.PythonFile(file, mode="r")
    return pyarrow.parquet.read_table(pyarrow_file)
