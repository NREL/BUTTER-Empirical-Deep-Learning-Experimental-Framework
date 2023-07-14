from functools import partial
from typing import Any
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import set_json_dumps
import simplejson
from uuid import UUID


def encode_complex(obj):
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(repr(obj) + " is not JSON serializable")


def json_dump_function(value: Any) -> str:
    return simplejson.dumps(
        value, sort_keys=True, separators=(",", ":"), default=encode_complex
    )


set_json_dumps(json_dump_function)


sql_comma = SQL(",")  # sql comma delimiter
sql_placeholder = SQL("%b")  # sql placeholders / references
