from typing import Dict, Iterable, Sequence, Tuple, Type, Union, List
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
                    array = array.astype(numpy.Int64)
                elif has_bool:
                    array = array.astype(numpy.B)

        return array

    return make_pyarrow_table_from_numpy(
        [str(column) for column in columns],
        [to_numpy(column) for column in columns],
    )


def make_dataframe_from_dict(data: Dict[str, Iterable]) -> pandas.DataFrame:
    cols = {}
    for name, col in data.items():
        cols[name] = numpy.array(col)
    return pandas.DataFrame(cols)


def make_pyarrow_table_from_numpy(
    columns: Sequence[str],
    numpy_arrays: Sequence[numpy.ndarray],
) -> Tuple[pyarrow.Table, List[str]]:
    schema, use_byte_stream_split = make_pyarrow_schema_from_dict(
        zip(columns, numpy_arrays)
    )  # type: ignore
    return pyarrow.Table.from_arrays(numpy_arrays, schema=schema), use_byte_stream_split


def make_pyarrow_schema_from_dict(
    columns: Iterable[Tuple[str, Union[list, ndarray]]],
) -> Tuple[pyarrow.Schema, List[str]]:
    fields = []
    use_byte_stream_split = []

    for name, values in columns:
        pyarrow_type, nullable, use_byte_stream_split_ = get_pyarrow_type_mapping(
            values
        )

        if pyarrow_type is None:
            continue

        if use_byte_stream_split_:
            use_byte_stream_split.append(name)

        fields.append(pyarrow.field(name, pyarrow_type, nullable=nullable))
    return pyarrow.schema(fields), use_byte_stream_split


def _check_type(t, x):
    return isinstance(t, Type) and issubclass(t, x)


def get_pyarrow_type_mapping(
    values: Union[list, ndarray],
) -> Tuple[pyarrow.DataType, bool, bool]:
    nullable = any((v is None for v in values))
    use_byte_stream_split = False

    t = None
    if isinstance(values, ndarray):
        t = values.dtype
    else:
        if len(values) == 0:
            return None, nullable, use_byte_stream_split
        t = type(values[0])

    def check_integer():
        nonlocal dst_type, nullable
        hi = max(filter(lambda v: v is not None, values))
        lo = min(filter(lambda v: v is not None, values))
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
        nullable = False
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
            return None, nullable, use_byte_stream_split
        else:
            dst_type = pyarrow.list_(get_pyarrow_type_mapping(element_type[0]))
    else:
        has_bool = False
        has_int = False
        has_float = False
        has_str = False
        has_null = False

        for v in values:
            has_bool |= isinstance(v, bool)
            has_int |= isinstance(v, int)
            has_float |= isinstance(v, float)
            has_str |= isinstance(v, str)
            has_null |= v is None

        nullable |= has_null

        if has_str:
            dst_type = pyarrow.string()
        elif has_float:
            dst_type = pyarrow.float32()
        elif has_int:
            check_integer()
        elif has_bool:
            dst_type = pyarrow.bool_()
        else:
            raise NotImplementedError(f"Unhandled type {t}.")

    return dst_type, nullable, use_byte_stream_split


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
