from ast import arg
from math import ceil
from psycopg import sql

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
    print("Connection object constructed.)
    try:
        with connection.cursor() as cursor:
            print("Cursor constructed.)
            cursor.execute(
                sql.SQL(
                    """
    select
        queue, min(priority) min_priority, max(priority) max_priority, avg(priority) avg_priority, command->'batch' batch, command->'shape' shape, status, count(*), avg(log((command->'size')::bigint)/log(2))::bigint avg_log_size
    from
        job_status s,
        job_data d
    where s.id = d.id and queue = 1
    group by status, queue, batch, shape
    order by status asc, min_priority asc, queue, batch, shape
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
