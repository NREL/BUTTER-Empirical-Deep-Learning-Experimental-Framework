from math import ceil
import random
import time
from typing import Any, Dict
from psycopg import sql
import psycopg

import jobqueue
from pprint import pprint
import sys

import dmp.postgres_interface


sys.path.append("../../")

q2 = """
UPDATE "run"
    SET
        "status" = 1,
        "worker_id" = '093bd5e3e9a94108a6553fac2cb9faa7'::uuid,
        "start_time" = NOW(),
        "update_time" = NOW()
    WHERE TRUE
        AND id IN (
            SELECT id FROM
                "run" "_run_selection"
            WHERE TRUE
                AND "_run_selection"."status" = 0
                AND "_run_selection"."queue" = 10
            ORDER BY "priority" DESC, "id" ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
    )
RETURNING
    "id","queue","status","priority","start_time","update_time","worker_id","parent_id","experiment_id","command","history","extended_history","error_message"

"""


q3 = """
UPDATE "run"
    SET
        "status" = 1,
        "worker_id" = '093bd5e3e9a94108a6553fac2cb9faa7'::uuid,
        "start_time" = NOW(),
        "update_time" = NOW()
    WHERE TRUE
        AND id IN (
            SELECT id FROM
                "run" "_run_selection"
            WHERE TRUE
                AND "_run_selection"."status" = 0
                AND "_run_selection"."queue" = 10
            ORDER BY "priority" DESC, "id" ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            )
RETURNING "id","queue","status","priority","start_time","update_time","worker_id","parent_id","experiment_id","error_message"
"""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=str, help="project name to test")
    args = parser.parse_args()
    project = args.project
    print(f'checking database connection for project "{project}"...')

    credentials = jobqueue.load_credentials(project)

    print(credentials)
    print("Credentials loaded, attempting to connect.")

    connection = jobqueue.connect(credentials)
    print("Connection object constructed.")
    try:
        with connection.cursor(binary=True) as cursor:
            print("Cursor constructed.")
            cursor.execute(q3, binary=True)
            # cursor.execute(
            #     sql.SQL(
            #         """
            # select
            #     queue, status, count(1) num
            # from
            #     run
            # where queue >= 0
            # group by queue, status
            # order by queue, status
            # ;"""
            #     )
            # )
            print("Query Executed.")
            for i in cursor.fetchall():
                print(i)
        print("All is well.")
    except Exception as e:
        print(f"Exception: f{e}")
    print("Done.")


if __name__ == "__main__":
    main()
