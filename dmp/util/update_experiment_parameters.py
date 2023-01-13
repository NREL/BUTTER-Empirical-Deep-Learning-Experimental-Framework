from math import ceil
import ujson
from psycopg import sql
import psycopg.extras as extras
import psycopg
import jobqueue.connect as connect
from pprint import pprint
import sys
from jobqueue.cursor_manager import CursorManager

from dmp.task.aspect_test.aspect_test_task import AspectTestTask

from dmp.logging.postgres_attribute_map import PostgresAttributeMap

sys.path.append("../../")

psycopg.extras.register_uuid()  # type: ignore

class ParameterUpdate:

    def __init__(self, credentials, ids) -> None:
        print('init ParameterUpdate')
        with CursorManager(credentials) as cursor:
            self.cursor = cursor
            self._parameter_map = PostgresAttributeMap(cursor)
            self._experiment_table = sql.Identifier("experiment")
            self._run_table = sql.Identifier("run")
            self._run_settings_table = sql.Identifier("run_settings")
            self._run_columns = [
                "seed",
                "save_every_epochs",
            ]

            """
            + run results:
                + update experiment parameters if not already updated
                + copy run data and new parameters into a new table 
                    with updated parameters
                or

                + update experiment table parameters
                + create new run_settings entry
                + for new runs:
                    + create run table entry
                        + use dummy data for other columns
                + for old runs:
                    + update run parameters
                
                + update materialized table
                
            """

            cursor.execute(sql.SQL("""
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
        r.parameters
    FROM jobqueue j, run r
    WHERE 
        r.job_id = j.uuid
        AND j.uuid IN %s
;"""), (ids,))
            rows = list(cursor.fetchall())
            values = [self.convert_row(row) for row in rows]

            values_sql = ','.join((
                cursor.mogrify(
                    '(' + (','.join(['%s' for _ in e])) + ')', e).decode("utf-8")
                for e in values))
            cursor.execute(
                sql.SQL("""
    WITH v as (
        SELECT
            job_id::uuid job_id,
            experiment_id,
            experiment_parameters::smallint[] experiment_parameters,
            run_parameters::smallint[] run_parameters,
            seed::bigint seed,
            save_every_epochs::smallint save_every_epochs
        FROM
            (VALUES """ + values_sql + """ ) AS t (
                job_id,
                experiment_id,
                experiment_parameters, 
                run_parameters,
                {run_columns}
                )
    ),
    x AS (
        SELECT experiment_id FROM {experiment_table} e
        WHERE e.experiment_id IN (SELECT distinct experiment_id from v)
        FOR UPDATE SKIP LOCKED
    )
    UPDATE {experiment_table} e SET
        parameters = v.experiment_parameters
    FROM 
        v
    WHERE
        e.experiment_id = v.experiment_id
        AND e.experiment_id IN (SELECT experiment_id from x)
    ;
    """
                        ).format(
                    run_columns=sql.SQL(',').join(
                        map(sql.Identifier, self._run_columns)),
                    experiment_table=self._experiment_table,
                    run_table=self._run_table,
                    run_settings_table=self._run_settings_table,
                ))
                
            cursor.execute(
                sql.SQL("""
    WITH v as (
        SELECT
            job_id::uuid job_id,
            experiment_id,
            experiment_parameters::smallint[] experiment_parameters,
            run_parameters::smallint[] run_parameters,
            seed::bigint seed,
            save_every_epochs::smallint save_every_epochs
        FROM
            (VALUES """ + values_sql + """ ) AS t (
                job_id,
                experiment_id,
                experiment_parameters, 
                run_parameters,
                {run_columns}
                )
    )
    INSERT INTO {run_settings_table} (
        job_id,
        experiment_id,
        parameters,
        {run_columns})
    SELECT
        job_id,
        experiment_id,
        run_parameters,
        {run_columns}
    FROM
        v
    ON CONFLICT DO NOTHING
    ;"""
                        ).format(
                    run_columns=sql.SQL(',').join(
                        map(sql.Identifier, self._run_columns)),
                    experiment_table=self._experiment_table,
                    run_table=self._run_table,
                    run_settings_table=self._run_settings_table,
                ))

    def convert_row(self, row):
        # print(f'{row}')
        # print(ujson.dumps(row[2], sort_keys=True, indent=2))

        config = row[2]
        kwargs = {
            'seed': config['seed'],
            'dataset': config['dataset'],
            'input_activation': config['activation'],
            'activation': config['activation'],
            'optimizer': config['optimizer'],
            'shape': config['topology'],
            'size': config['budget'],
            'depth': config['depth'],
            'validation_split': config['run_config']['validation_split'],
            'validation_split_method': config['validation_split_method'],
            'run_config': config['run_config'],
            'label_noise': config['label_noise'],
            'early_stopping': config.get('early_stopping', None),
            'save_every_epochs': None,
        }

        kwargs['batch'] = 'fixed_3k_1'

        if not isinstance(kwargs['early_stopping'], dict):
            kwargs['early_stopping'] = None

        if kwargs.get('residual_mode', None) == 'full':
            kwargs['shape'] = kwargs['shape'] + '_' + 'residual'
        
        if kwargs['label_noise'] == 'none':
            kwargs['label_noise'] = 0.0

        task = AspectTestTask(**kwargs)

        # pprint(task.extract_parameters())

        experiment_parameters, run_parameters, run_values = \
            task.parameters

        run_parameters['tensorflow_version'] = '2.6.0'
        run_parameters.update(experiment_parameters)

        job_id = row[0]

        # print(job_id)
        # pprint(experiment_parameters)
        # pprint(run_parameters)
        # pprint(run_values)

        run_parameter_ids = \
            self._parameter_map.to_sorted_parameter_ids(
                run_parameters,
                self.cursor
            )

        experiment_parameter_ids = \
            self._parameter_map.to_sorted_parameter_ids(
                experiment_parameters,
                self.cursor
            )
        # pprint(run_parameter_ids)
        # pprint(experiment_parameter_ids)

        # result_columns = sorted(list(result.keys()))
        return (job_id,
                row[16],
                experiment_parameter_ids,
                run_parameter_ids,
                *(run_values[c] for c in self._run_columns))


if __name__ == "__main__":
    credentials = connect.load_credentials('dmp')

    extras.register_default_json(loads=ujson.loads, globally=True)
    extras.register_default_jsonb(loads=ujson.loads, globally=True)

    ids = None
    with CursorManager(credentials) as cursor:
        cursor.execute(sql.SQL("""
    SELECT 
        j.uuid
    FROM jobqueue j
    WHERE j.groupname = 'fixed_3k_1'
    AND j.status = 'done'
    AND EXISTS (SELECT job_id FROM run WHERE run.job_id = j.uuid)
    AND NOT EXISTS (SELECT job_id FROM run_settings s WHERE s.job_id = j.uuid)
;"""))
        ids = [row[0] for row in cursor.fetchall()]

    print(f'loaded {len(ids)}.')

    from pathos.multiprocessing import Pool
    num_readers = 32
    chunk_size = 8192
    chunks = [
        tuple(ids[c*chunk_size: min(len(ids), (c+1)*chunk_size)])
        for c in range(int(ceil(len(ids)/chunk_size)))]
    print(f'created {len(chunks)} chunks')
    # ParameterUpdate(credentials, chunks[0])
    def func(c):
        ParameterUpdate(credentials, c)
        return None

    with Pool(num_readers) as p:
        p.map(func, chunks)
