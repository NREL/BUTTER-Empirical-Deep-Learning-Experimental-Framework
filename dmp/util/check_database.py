from ast import arg
from math import ceil
from psycopg2 import sql
import psycopg2.extras as extras
import psycopg2
import jobqueue.connect as connect
from pprint import pprint
import sys
import jobqueue.connect

sys.path.append("../../")

psycopg2.extras.register_uuid()

def main():
    import simplejson
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=str,
                        help='project name to test')
    args = parser.parse_args()
    project = args.project
    print(f'checking database connection for project "{project}"...')
    
    credentials = connect.load_credentials(project)

    extras.register_default_json(loads=simplejson.loads, globally=True)
    extras.register_default_jsonb(loads=simplejson.loads, globally=True)
    psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)
    
    print(credentials)
    print('Credentials loaded, attempting to connect.')
    connection = connect.connect(credentials)
    try:
        cursor = connection.cursor()
        cursor.execute(sql.SQL("""
select 
    queue, min(priority) min_priority, max(priority) max_priority, avg(priority) avg_priority, command->'batch' batch, command->'shape' shape, status, count(*), avg(log((command->'size')::bigint)/log(2))::bigint avg_log_size
from 
    job_status s,
    job_data d
where s.id = d.id and queue = 1
group by status, queue, batch, shape
order by status asc, min_priority asc, queue, batch, shape
;"""))
        for i in cursor.fetchall():
            print(i)
        print("All is well.")
    except Exception as e:
        print('Exception: f{e}')
    print("Done.")

if __name__ == "__main__":
    main()
