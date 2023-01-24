from typing import Any
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import set_json_dumps
import simplejson

def json_dump_function(value: Any) -> str:
    return simplejson.dumps(value, sort_keys=True, separators=(',', ':'))


set_json_dumps(json_dump_function)

comma_sql = SQL(',')  # sql comma delimiter
placeholder_sql = SQL('%s')  # sql placeholders / references