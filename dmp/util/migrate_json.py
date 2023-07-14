import hashlib
import os
from jobqueue.connection_manager import ConnectionManager
import psycopg
import simplejson

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import uuid
from psycopg import sql

from jobqueue import load_credentials


def json_dump_function(value):
    return simplejson.dumps(value, sort_keys=True, separators=(",", ":"))


def _make_json_digest(value) -> uuid.UUID:
    return uuid.UUID(hashlib.md5(json_dump_function(value).encode("utf-8")).hexdigest())


def _make_json_digest_dumped(dumped) -> uuid.UUID:
    return uuid.UUID(hashlib.md5(dumped.encode("utf-8")).hexdigest())


psycopg.types.json.set_json_dumps(json_dump_function)
credentials = load_credentials("dmp")
with ConnectionManager(credentials) as connection:
    while True:
        with connection.cursor(binary=True) as cursor:
            cursor.execute(
                sql.SQL(
                    """
            SELECT attr_id, value_json FROM attr WHERE value_json IS NOT NULL AND digest IS NULL AND value_type = 5 LIMIT 8192;
            """
                )
            )

            vals = [
                (
                    i,
                    _make_json_digest(v),
                    psycopg.types.json.Jsonb(v, dumps=json_dump_function),
                )
                for i, v in (
                    (row[0], simplejson.loads(json_dump_function(row[1])))
                    for row in cursor
                )
            ]

            if len(vals) == 0:
                break

            p = sql.SQL("(%s,%s,%s)")
            value_placeholders = sql.SQL(",").join((p for v in vals))
            values = []
            for v in vals:
                values.extend(v)
                # print(v[2])

            connection.execute(
                sql.SQL(
                    """
UPDATE attr
    SET digest = v.digest, value_json = v.value_json
FROM (VALUES {}) AS v (attr_id, digest, value_json)
WHERE attr.attr_id = v.attr_id;"""
                ).format(value_placeholders),
                values,
                binary=True,
            )
