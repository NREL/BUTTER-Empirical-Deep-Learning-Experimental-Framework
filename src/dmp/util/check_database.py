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


def _extract_inner_credentials(credentials: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: credentials[k]
        for k in ("host", "port", "dbname", "user", "password")
        if k in credentials
    }


def connect(credentials: Dict[str, Any], autocommit: bool = True) -> Any:
    connection = None
    initial_wait_max = credentials.get("initial_wait_max", 12)
    min_wait = credentials.get("min_wait", 0.5)
    max_wait = credentials.get("max_wait", 2 * 60 * 60)
    max_attempts = credentials.get("max_attempts", 10000)
    attempts = 0
    while attempts < max_attempts:
        wait_time = 0.0
        try:
            inner_credentials = _extract_inner_credentials(credentials)
            connection_string = " ".join(
                (f"{key}={value}" for key, value in inner_credentials.items())
            )
            print(f"connecting: {connection_string}...")
            connection = psycopg.connect(connection_string)
            print(f"connected.")
            break
        except psycopg.OperationalError as e:
            print(f"OperationalError while connecting to database: {e}", flush=True)

            sleep_time = random.uniform(min_wait, max(initial_wait_max, wait_time))
            wait_time += sleep_time
            attempts += 1

            if attempts >= max_attempts or wait_time >= max_wait:
                raise e
            time.sleep(sleep_time)

    connection.autocommit = autocommit
    return connection


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

    connection = connect(credentials)
    print("Connection object constructed.")
    try:
        with connection.cursor() as cursor:
            print("Cursor constructed.")
            cursor.execute(
                sql.SQL(
                    """
            select
                queue, count(1) num
            from
                run
            where queue >= 0
            group by queue, status
            order by queue, status
            ;"""
                )
            )
            print("Query Executed.")
            for i in cursor.fetchall():
                print(i)
        print("All is well.")
    except Exception as e:
        print("Exception: f{e}")
    print("Done.")


if __name__ == "__main__":
    main()
