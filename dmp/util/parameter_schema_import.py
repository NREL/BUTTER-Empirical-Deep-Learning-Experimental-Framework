from pprint import pprint
import sys
from jobqueue.cursor_manager import CursorManager

from dmp.task.aspect_test.aspect_test_task import AspectTestTask



sys.path.append("../../")


import jobqueue.connect as connect

import psycopg2
import psycopg2.extras as extras
from psycopg2 import sql
import ujson


credentials = None

def parameter_schema_import():
    with CursorManager(credentials) as cursor:
        """
        + run results:
            + update experiment parameters if not already updated
            + copy run data and new parameters into a new table 
                with updated parameters
            or

            + update experiment parameters
            + update run parameters
            + wait until schema frees up to :
                + add run_parameters, save_every_epochs to run table
            
        + new results:
            + create/get experiment id
            + write data to run table
            
        """


        # query = sql.SQL("""
        # SELECT timestamp, doc, job, groupname 
        # FROM log 
        # WHERE groupname = 'fixed_3k_1'
        # LIMIT 10
        # """)
        query = sql.SQL("""
        SELECT 
            j.uuid,
            j.username,
            j.config,
            j.groupname,
            j.host,
            j.status,
            j.worker,
            j.creation_time,
            j.priority,
            j.start_time,
            j.update_time,
            j.end_time,
            j.depth,
            j.wall_time,
            j.retry_count,
            j.jobid,
            r.experiment_id,
            r.parameters,
            
        FROM jobqueue j, run r
        WHERE j.groupname = 'fixed_3k_1'
        AND j.status = 'done'
        AND r.job_id = j.uuid
        LIMIT 10
        """)
        cursor.execute(query)
        for row in cursor.fetchall():
            print(f'{row}')
            print(ujson.dumps(row[2], sort_keys=True, indent=2))
            
            config = row[2]
            kwargs = {
                'seed': config['seed'],
                'dataset' : config['dataset'],
                'input_activation' : config['activation'],
                'activation' : config['activation'],
                'optimizer' : config['optimizer'],
                'shape' : config['topology'],
                'size' : config['budget'],
                'depth' : config['depth'],
                'validation_split' : config['run_config']['validation_split'],
                'validation_split_method' : config['validation_split_method'],
                'run_config' : config['run_config'],
                'label_noise' : config['label_noise'],
                'early_stopping' : config.get('early_stopping', None),
                'save_every_epochs' : None,
            }

            kwargs['batch'] = 'fixed_3k_1'

            if not isinstance(kwargs['early_stopping'], dict):
                kwargs['early_stopping'] = None
            
            if kwargs.get('residual_mode', None) == 'full':
                kwargs['shape'] = kwargs['shape'] + '_' + 'residual'
            task = AspectTestTask(**kwargs)

            pprint(task.extract_parameters())

            experiment_parameters, run_parameters, run_values = \
                task.parameters

            run_parameters['tensorflow_version'] = '2.6.0'

            pprint(experiment_parameters)
            pprint(run_parameters)
            pprint(run_values)

        

        # tasks.append()



if __name__ == "__main__":
    credentials = connect.load_credentials('dmp')

    extras.register_default_json(loads=ujson.loads, globally=True)
    extras.register_default_jsonb(loads=ujson.loads, globally=True)

    parameter_schema_import()