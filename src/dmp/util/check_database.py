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
