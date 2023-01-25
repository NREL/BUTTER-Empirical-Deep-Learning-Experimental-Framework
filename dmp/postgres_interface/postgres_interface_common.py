from typing import Any
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import set_json_dumps
import simplejson

def json_dump_function(value: Any) -> str:
    return simplejson.dumps(value, sort_keys=True, separators=(',', ':'))


set_json_dumps(json_dump_function)

sql_comma = SQL(',')  # sql comma delimiter
sql_placeholder = SQL('%s')  # sql placeholders / references