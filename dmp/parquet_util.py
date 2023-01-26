from typing import Dict, Iterable, Sequence, Tuple, Type, Union, List
import pandas
import pyarrow
from numpy import ndarray
import numpy


def make_pyarrow_schema_from_panads(
    dataframe: pandas.DataFrame, ) -> Tuple[pyarrow.Schema, List[str]]:
    return make_pyarrow_schema_from_dict([(str(column),
                                           dataframe[column].to_numpy())
                                          for column in dataframe.columns])


def make_pyarrow_schema_from_dict(
    columns: Iterable[Tuple[str, Union[list, ndarray]]],
) -> Tuple[pyarrow.Schema, List[str]]:

    fields = []
    use_byte_stream_split = []

    for name, values in columns:
        pyarrow_type = get_pyarrow_type_mapping(values)
        if pyarrow_type is None:
            continue

        if pyarrow_type == pyarrow.float16()\
            or pyarrow_type == pyarrow.float32()\
            or pyarrow_type == pyarrow.float64():
            use_byte_stream_split.append(name)

        fields.append(pyarrow.field(name, pyarrow_type))
    return pyarrow.schema(fields), use_byte_stream_split


def _check_type(t, x):
    return (isinstance(t, Type) and issubclass(t, x))


def get_pyarrow_type_mapping(
    values: Union[list, ndarray], ) -> pyarrow.DataType:

    t = None
    if isinstance(values, ndarray):
        t = values.dtype
    else:
        if len(values) == 0:
            return None
        t = type(values[0])

    if _check_type(t, bool) or numpy.issubdtype(t, bool):
        return pyarrow.bool_()
    elif _check_type(t, int) or numpy.issubdtype(t, numpy.integer):
        hi = numpy.max(values)
        lo = numpy.min(values)
        if hi <= (2**7 - 1) and lo >= (-2**7):
            return pyarrow.int8()
        if hi <= (2**15 - 1) and lo >= (-2**15):
            return pyarrow.int16()
        if hi <= (2**31 - 1) and lo >= (-2**31):
            return pyarrow.int32()
        return pyarrow.int64()
    elif _check_type(t, float) or numpy.issubdtype(t, numpy.floating):
        return pyarrow.float32()
    elif _check_type(t, str)\
        or numpy.issubdtype(t, numpy.string_)\
        or numpy.issubdtype(t, numpy.str_):
        return pyarrow.string
    elif _check_type(t, list):
        element_type = next((et for et in (get_pyarrow_type_mapping(v)
                                           for v in values) if et is not None),
                            None)
        if element_type is None:
            return None
        return pyarrow.list_(get_pyarrow_type_mapping(element_type))

    raise NotImplementedError(f'Unhandled type {t}.')


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
        raise NotImplementedError(f'Unsupported dtype {source_type}.')

    significand, exponent = numpy.frexp(source)
    significand = numpy.bitwise_and(
        significand.view(int_type),
        numpy.bitwise_not(
            numpy.array([int(2 << bits_to_trim) - 1],
                        dtype=int_type))).view(source_type)
    return numpy.ldexp(significand, exponent)
