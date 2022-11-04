from ast import arg
from math import ceil
from jobqueue.cursor_manager import CursorManager
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
                        help='project name to use')

    args = parser.parse_args()
    project = args.project
    print(f'checking database connection for project "{project}"...')

    credentials = connect.load_credentials(project)

    extras.register_default_json(loads=simplejson.loads, globally=True)
    extras.register_default_jsonb(loads=simplejson.loads, globally=True)
    psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)

    print(credentials)
    print('Credentials loaded, attempting to connect.')
    with CursorManager(credentials) as cursor:
        while True:
            cursor.execute(sql.SQL("""
insert into experiment_summary_ (
    experiment_id,
    experiment_parameters,
    widths,
    network_structure,
    "size",
    num_free_parameters,
    relative_size_error,
    num_runs, num,
    test_loss_num_finite, test_loss_min, test_loss_max,
    test_loss_avg, test_loss_stddev,    
    test_loss_q1, test_loss_median, test_loss_q3,
    loss_num_finite, loss_min, loss_max,
    loss_avg, loss_stddev,
    loss_q1, loss_median, loss_q3,
    test_accuracy_avg, test_accuracy_stddev,
    test_accuracy_q1, test_accuracy_median, test_accuracy_q3,
    accuracy_avg, accuracy_stddev,
    accuracy_q1, accuracy_median, accuracy_q3,
    test_mean_squared_error_avg, test_mean_squared_error_stddev,
    test_mean_squared_error_q1, test_mean_squared_error_median, test_mean_squared_error_q3,
    mean_squared_error_avg, mean_squared_error_stddev,
    mean_squared_error_q1, mean_squared_error_median, mean_squared_error_q3,
    test_kullback_leibler_divergence_avg, test_kullback_leibler_divergence_stddev,
    test_kullback_leibler_divergence_q1, test_kullback_leibler_divergence_median, test_kullback_leibler_divergence_q3,
    kullback_leibler_divergence_avg, kullback_leibler_divergence_stddev,
    kullback_leibler_divergence_q1, kullback_leibler_divergence_median, kullback_leibler_divergence_q3,
    test_loss_min_epoch_q1, test_loss_min_epoch_median, test_loss_min_epoch_q3, test_loss_min_value_min, test_loss_min_value_max, test_loss_min_value_avg, test_loss_min_value_q1, test_loss_min_value_median, test_loss_min_value_q3, test_accuracy_max_epoch_min, test_accuracy_max_epoch_max, test_accuracy_max_epoch_avg, test_accuracy_max_epoch_q1, test_accuracy_max_epoch_median, test_accuracy_max_epoch_q3, test_accuracy_max_value_min, test_accuracy_max_value_max, test_accuracy_max_value_avg, test_accuracy_max_value_q1, test_accuracy_max_value_median, test_accuracy_max_value_q3, test_mean_squared_error_min_epoch_min, test_mean_squared_error_min_epoch_max, test_mean_squared_error_min_epoch_avg, test_mean_squared_error_min_epoch_q1, test_mean_squared_error_min_epoch_median, test_mean_squared_error_min_epoch_q3, test_mean_squared_error_min_value_min, test_mean_squared_error_min_value_max, test_mean_squared_error_min_value_avg, test_mean_squared_error_min_value_q1, test_mean_squared_error_min_value_median, test_mean_squared_error_min_value_q3, test_kullback_leibler_divergence_min_epoch_min, test_kullback_leibler_divergence_min_epoch_max, test_kullback_leibler_divergence_min_epoch_avg, test_kullback_leibler_divergence_min_epoch_q1, test_kullback_leibler_divergence_min_epoch_median, test_kullback_leibler_divergence_min_epoch_q3, test_kullback_leibler_divergence_min_value_min, test_kullback_leibler_divergence_min_value_max, test_kullback_leibler_divergence_min_value_avg, test_kullback_leibler_divergence_min_value_q1, test_kullback_leibler_divergence_min_value_median, test_kullback_leibler_divergence_min_value_q3
)
select
    e.experiment_id,
    experiment_parameters,
    widths,
    network_structure,
    e.size,
    e.num_free_parameters,
    e.relative_size_error,
    num_runs, num,
    test_loss_num_finite, test_loss_min, test_loss_max,
    test_loss_avg, test_loss_stddev,    
    test_loss_q1, test_loss_median, test_loss_q3,
    loss_num_finite, loss_min, loss_max,
    loss_avg, loss_stddev,
    loss_q1, loss_median, loss_q3,
    test_accuracy_avg, test_accuracy_stddev,
    test_accuracy_q1, test_accuracy_median, test_accuracy_q3,
    accuracy_avg, accuracy_stddev,
    accuracy_q1, accuracy_median, accuracy_q3,
    test_mean_squared_error_avg, test_mean_squared_error_stddev,
    test_mean_squared_error_q1, test_mean_squared_error_median, test_mean_squared_error_q3,
    mean_squared_error_avg, mean_squared_error_stddev,
    mean_squared_error_q1, mean_squared_error_median, mean_squared_error_q3,
    test_kullback_leibler_divergence_avg, test_kullback_leibler_divergence_stddev,
    test_kullback_leibler_divergence_q1, test_kullback_leibler_divergence_median, test_kullback_leibler_divergence_q3,
    kullback_leibler_divergence_avg, kullback_leibler_divergence_stddev,
    kullback_leibler_divergence_q1, kullback_leibler_divergence_median, kullback_leibler_divergence_q3,
    test_loss_min_epoch_q1, test_loss_min_epoch_median, test_loss_min_epoch_q3, test_loss_min_value_min, test_loss_min_value_max, test_loss_min_value_avg, test_loss_min_value_q1, test_loss_min_value_median, test_loss_min_value_q3, test_accuracy_max_epoch_min, test_accuracy_max_epoch_max, test_accuracy_max_epoch_avg, test_accuracy_max_epoch_q1, test_accuracy_max_epoch_median, test_accuracy_max_epoch_q3, test_accuracy_max_value_min, test_accuracy_max_value_max, test_accuracy_max_value_avg, test_accuracy_max_value_q1, test_accuracy_max_value_median, test_accuracy_max_value_q3, test_mean_squared_error_min_epoch_min, test_mean_squared_error_min_epoch_max, test_mean_squared_error_min_epoch_avg, test_mean_squared_error_min_epoch_q1, test_mean_squared_error_min_epoch_median, test_mean_squared_error_min_epoch_q3, test_mean_squared_error_min_value_min, test_mean_squared_error_min_value_max, test_mean_squared_error_min_value_avg, test_mean_squared_error_min_value_q1, test_mean_squared_error_min_value_median, test_mean_squared_error_min_value_q3, test_kullback_leibler_divergence_min_epoch_min, test_kullback_leibler_divergence_min_epoch_max, test_kullback_leibler_divergence_min_epoch_avg, test_kullback_leibler_divergence_min_epoch_q1, test_kullback_leibler_divergence_min_epoch_median, test_kullback_leibler_divergence_min_epoch_q3, test_kullback_leibler_divergence_min_value_min, test_kullback_leibler_divergence_min_value_max, test_kullback_leibler_divergence_min_value_avg, test_kullback_leibler_divergence_min_value_q1, test_kullback_leibler_divergence_min_value_median, test_kullback_leibler_divergence_min_value_q3
from
(
    select
        e.*
    from 
        experiment_ e 
    where
        e.experiment_id IN (select distinct r.experiment_id from run_ r where not exists (
            select * from experiment_summary_ s where 
                s.experiment_id = r.experiment_id and 
                    (s.update_timestamp > r.record_timestamp)
        ))
    limit 128
    for update
    skip locked
) e,
lateral (
    with rr as (
        select run_.ctid as run_ctid from run_ where run_.experiment_id = e.experiment_id
    )
    select * from
        (
            with ev as (
                select run_ctid, v, epoch from
                    rr inner join run_ r on rr.run_ctid = r.ctid,
                    unnest(r.test_loss) WITH ORDINALITY as epoch_value(v, epoch)
            )
            select * from
            (
                select
                    max(v_num) num_runs, array_agg(v_num) num,
                    array_agg(v_num_finite) test_loss_num_finite, array_agg(v_min) test_loss_min, array_agg(v_max) test_loss_max,    
                    array_agg(v_avg) test_loss_avg, array_agg(v_stddev) test_loss_stddev,                
                    array_agg(v_percentile[1]) test_loss_q1, array_agg(v_percentile[2]) test_loss_median, array_agg(v_percentile[3]) test_loss_q3
                from
                (       
                    select
                        (COUNT(coalesce(v, 'NaN'::real)))::smallint v_num, 
                        (COUNT(v))::smallint v_num_finite, MIN(v)::real v_min, MAX(v)::real v_max,    
                        (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev,
                        (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                    from
                        ev
                    group by epoch order by epoch
                ) x
            ) x,
            (
                select
                    test_loss_min_epoch_min, test_loss_min_epoch_max, test_loss_min_epoch_avg, test_loss_min_epoch_percentile[1] test_loss_min_epoch_q1, test_loss_min_epoch_percentile[2] test_loss_min_epoch_median, test_loss_min_epoch_percentile[3] test_loss_min_epoch_q3, 
                    test_loss_min_value_min, test_loss_min_value_max, test_loss_min_value_avg, test_loss_min_value_percentile[1] test_loss_min_value_q1, test_loss_min_value_percentile[2] test_loss_min_value_median, test_loss_min_value_percentile[3] test_loss_min_value_q3
                from
                    (
                    select
                        min(epoch)::integer test_loss_min_epoch_min, max(epoch)::integer test_loss_min_epoch_max, avg(epoch)::real test_loss_min_epoch_avg, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(epoch, 'NaN'::real)))::real[] test_loss_min_epoch_percentile,
                        min(v)::real test_loss_min_value_min, max(v)::real test_loss_min_value_max, avg(v)::real test_loss_min_value_avg, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] test_loss_min_value_percentile
                    from
                        (
                        select
                            distinct on (run_ctid) run_ctid, epoch, v
                        from
                            ev
                        where v is not null and epoch is not null
                        order by run_ctid, v asc, epoch asc
                        ) x
                    ) x
            ) y
        ) test_loss_,
    lateral (
        select
            array_agg(v_num_finite) loss_num_finite, array_agg(v_min) loss_min, array_agg(v_max) loss_max,
            array_agg(v_avg) loss_avg, array_agg(v_stddev) loss_stddev,
            array_agg(v_percentile[1]) loss_q1, array_agg(v_percentile[2]) loss_median, array_agg(v_percentile[3]) loss_q3
        from
        (       
            select
                (COUNT(v))::smallint v_num_finite, MIN(v)::real v_min, MAX(v)::real v_max,    
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev,
                (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
            from
                rr inner join run_ r on rr.run_ctid = r.ctid,
                unnest(r.loss) WITH ORDINALITY as epoch_value(v, epoch)
            group by epoch order by epoch
        ) x
    ) loss_,
    lateral (
        with ev as (
            select run_ctid, v, epoch from
                rr inner join run_ r on rr.run_ctid = r.ctid,
                unnest(r.test_accuracy) WITH ORDINALITY as epoch_value(v, epoch)
        )
        select * from
            (
            select array_agg(v_avg) test_accuracy_avg, array_agg(v_stddev) test_accuracy_stddev, array_agg(v_percentile[1]) test_accuracy_q1, array_agg(v_percentile[2]) test_accuracy_median, array_agg(v_percentile[3]) test_accuracy_q3 from
                (select
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                    from ev
                    group by epoch order by epoch
            ) x ) x,
            (
                select
                    test_accuracy_max_epoch_min, test_accuracy_max_epoch_max, test_accuracy_max_epoch_avg, test_accuracy_max_epoch_percentile[1] test_accuracy_max_epoch_q1, test_accuracy_max_epoch_percentile[2] test_accuracy_max_epoch_median, test_accuracy_max_epoch_percentile[3] test_accuracy_max_epoch_q3, 
                    test_accuracy_max_value_min, test_accuracy_max_value_max, test_accuracy_max_value_avg, test_accuracy_max_value_percentile[1] test_accuracy_max_value_q1, test_accuracy_max_value_percentile[2] test_accuracy_max_value_median, test_accuracy_max_value_percentile[3] test_accuracy_max_value_q3
                from
                    (
                    select
                        min(epoch)::integer test_accuracy_max_epoch_min, max(epoch)::integer test_accuracy_max_epoch_max, avg(epoch)::real test_accuracy_max_epoch_avg, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(epoch, 'NaN'::real)))::real[] test_accuracy_max_epoch_percentile,
                        min(v)::real test_accuracy_max_value_min, max(v)::real test_accuracy_max_value_max, avg(v)::real test_accuracy_max_value_avg, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] test_accuracy_max_value_percentile
                    from
                        (
                            select
                                distinct on (run_ctid) run_ctid, epoch, v
                            from
                                ev
                            where v is not null and epoch is not null
                            order by run_ctid, v desc, epoch asc
                        ) x
                    ) x
            ) y
    ) test_accuracy_,
    lateral (select array_agg(v_avg) accuracy_avg, array_agg(v_stddev) accuracy_stddev, array_agg(v_percentile[1]) accuracy_q1, array_agg(v_percentile[2]) accuracy_median, array_agg(v_percentile[3]) accuracy_q3 from
            (select
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.accuracy) WITH ORDINALITY as epoch_value(v, epoch)
                group by epoch order by epoch
            ) x ) accuracy_,
    lateral (
        with ev as (
            select run_ctid, v, epoch from
                rr inner join run_ r on rr.run_ctid = r.ctid,
                unnest(r.test_mean_squared_error) WITH ORDINALITY as epoch_value(v, epoch)
        )
        select * from 
            (
            select array_agg(v_avg) test_mean_squared_error_avg, array_agg(v_stddev) test_mean_squared_error_stddev, array_agg(v_percentile[1]) test_mean_squared_error_q1, array_agg(v_percentile[2]) test_mean_squared_error_median, array_agg(v_percentile[3]) test_mean_squared_error_q3 from
                ( select
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                    from ev
                    group by epoch order by epoch
            ) x ) x,
            (
                select
                    test_mean_squared_error_min_epoch_min, test_mean_squared_error_min_epoch_max, test_mean_squared_error_min_epoch_avg, test_mean_squared_error_min_epoch_percentile[1] test_mean_squared_error_min_epoch_q1, test_mean_squared_error_min_epoch_percentile[2] test_mean_squared_error_min_epoch_median, test_mean_squared_error_min_epoch_percentile[3] test_mean_squared_error_min_epoch_q3, 
                    test_mean_squared_error_min_value_min, test_mean_squared_error_min_value_max, test_mean_squared_error_min_value_avg, test_mean_squared_error_min_value_percentile[1] test_mean_squared_error_min_value_q1, test_mean_squared_error_min_value_percentile[2] test_mean_squared_error_min_value_median, test_mean_squared_error_min_value_percentile[3] test_mean_squared_error_min_value_q3
                from
                    (
                    select
                        min(epoch)::integer test_mean_squared_error_min_epoch_min, max(epoch)::integer test_mean_squared_error_min_epoch_max, avg(epoch)::real test_mean_squared_error_min_epoch_avg, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(epoch, 'NaN'::real)))::real[] test_mean_squared_error_min_epoch_percentile,
                        min(v)::real test_mean_squared_error_min_value_min, max(v)::real test_mean_squared_error_min_value_max, avg(v)::real test_mean_squared_error_min_value_avg, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] test_mean_squared_error_min_value_percentile
                    from
                        (
                            select
                                distinct on (run_ctid) run_ctid, epoch, v
                            from
                                ev
                            where v is not null and epoch is not null
                            order by run_ctid, v asc, epoch asc
            ) x ) x ) y
        ) test_mean_squared_error_,
    lateral (select array_agg(v_avg) mean_squared_error_avg, array_agg(v_stddev) mean_squared_error_stddev, array_agg(v_percentile[1]) mean_squared_error_q1, array_agg(v_percentile[2]) mean_squared_error_median, array_agg(v_percentile[3]) mean_squared_error_q3 from
            (select
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.mean_squared_error) WITH ORDINALITY as epoch_value(v, epoch)
                group by epoch order by epoch
            ) x ) mean_squared_error_,
    lateral (
        with ev as (
            select run_ctid, v, epoch from
                rr inner join run_ r on rr.run_ctid = r.ctid,
                unnest(r.test_kullback_leibler_divergence) WITH ORDINALITY as epoch_value(v, epoch)
        )
        select * from 
            (
        select array_agg(v_avg) test_kullback_leibler_divergence_avg, array_agg(v_stddev) test_kullback_leibler_divergence_stddev, array_agg(v_percentile[1]) test_kullback_leibler_divergence_q1, array_agg(v_percentile[2]) test_kullback_leibler_divergence_median, array_agg(v_percentile[3]) test_kullback_leibler_divergence_q3 from
            (select
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                from ev
                group by epoch order by epoch
            ) x ) x,
        (select
                    test_kullback_leibler_divergence_min_epoch_min, test_kullback_leibler_divergence_min_epoch_max, test_kullback_leibler_divergence_min_epoch_avg, test_kullback_leibler_divergence_min_epoch_percentile[1] test_kullback_leibler_divergence_min_epoch_q1, test_kullback_leibler_divergence_min_epoch_percentile[2] test_kullback_leibler_divergence_min_epoch_median, test_kullback_leibler_divergence_min_epoch_percentile[3] test_kullback_leibler_divergence_min_epoch_q3, 
                    test_kullback_leibler_divergence_min_value_min, test_kullback_leibler_divergence_min_value_max, test_kullback_leibler_divergence_min_value_avg, test_kullback_leibler_divergence_min_value_percentile[1] test_kullback_leibler_divergence_min_value_q1, test_kullback_leibler_divergence_min_value_percentile[2] test_kullback_leibler_divergence_min_value_median, test_kullback_leibler_divergence_min_value_percentile[3] test_kullback_leibler_divergence_min_value_q3
                from
                    (
                    select
                        min(epoch)::integer test_kullback_leibler_divergence_min_epoch_min, max(epoch)::integer test_kullback_leibler_divergence_min_epoch_max, avg(epoch)::real test_kullback_leibler_divergence_min_epoch_avg, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(epoch, 'NaN'::real)))::real[] test_kullback_leibler_divergence_min_epoch_percentile,
                        min(v)::real test_kullback_leibler_divergence_min_value_min, max(v)::real test_kullback_leibler_divergence_min_value_max, avg(v)::real test_kullback_leibler_divergence_min_value_avg, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] test_kullback_leibler_divergence_min_value_percentile
                    from
                        (
                            select
                                distinct on (run_ctid) run_ctid, epoch, v
                            from
                                ev
                            where v is not null and epoch is not null
                            order by run_ctid, v asc, epoch asc
    ) x ) x ) y ) test_kullback_leibler_divergence_,
    lateral (select array_agg(v_avg) kullback_leibler_divergence_avg, array_agg(v_stddev) kullback_leibler_divergence_stddev, array_agg(v_percentile[1]) kullback_leibler_divergence_q1, array_agg(v_percentile[2]) kullback_leibler_divergence_median, array_agg(v_percentile[3]) kullback_leibler_divergence_q3 from
            (select
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.kullback_leibler_divergence) WITH ORDINALITY as epoch_value(v, epoch)
                group by epoch order by epoch
            ) x ) kullback_leibler_divergence_
    ) aggregates
ON CONFLICT (experiment_id) DO UPDATE SET
        update_timestamp = ((date_part('epoch'::text, CURRENT_TIMESTAMP) - (1600000000)::double precision))::integer,
        widths = EXCLUDED.widths,
        network_structure = EXCLUDED.network_structure,
        "size" = EXCLUDED."size",
        num_free_parameters = EXCLUDED.num_free_parameters,
        relative_size_error = EXCLUDED.relative_size_error,
        num_runs = EXCLUDED.num_runs,num = EXCLUDED.num,
        test_loss_num_finite = EXCLUDED.test_loss_num_finite,test_loss_min = EXCLUDED.test_loss_min,test_loss_max = EXCLUDED.test_loss_max,
        test_loss_avg = EXCLUDED.test_loss_avg,test_loss_stddev = EXCLUDED.test_loss_stddev,    
        test_loss_q1 = EXCLUDED.test_loss_q1,test_loss_median = EXCLUDED.test_loss_median,test_loss_q3 = EXCLUDED.test_loss_q3,
        loss_num_finite = EXCLUDED.loss_num_finite,loss_min = EXCLUDED.loss_min,loss_max = EXCLUDED.loss_max,
        loss_avg = EXCLUDED.loss_avg,loss_stddev = EXCLUDED.loss_stddev,
        loss_q1 = EXCLUDED.loss_q1,loss_median = EXCLUDED.loss_median,loss_q3 = EXCLUDED.loss_q3,
        test_accuracy_avg = EXCLUDED.test_accuracy_avg,test_accuracy_stddev = EXCLUDED.test_accuracy_stddev,
        test_accuracy_q1 = EXCLUDED.test_accuracy_q1,test_accuracy_median = EXCLUDED.test_accuracy_median,test_accuracy_q3 = EXCLUDED.test_accuracy_q3,
        accuracy_avg = EXCLUDED.accuracy_avg,accuracy_stddev = EXCLUDED.accuracy_stddev,
        accuracy_q1 = EXCLUDED.accuracy_q1,accuracy_median = EXCLUDED.accuracy_median,accuracy_q3 = EXCLUDED.accuracy_q3,
        test_mean_squared_error_avg = EXCLUDED.test_mean_squared_error_avg,test_mean_squared_error_stddev = EXCLUDED.test_mean_squared_error_stddev,
        test_mean_squared_error_q1 = EXCLUDED.test_mean_squared_error_q1,test_mean_squared_error_median = EXCLUDED.test_mean_squared_error_median,test_mean_squared_error_q3 = EXCLUDED.test_mean_squared_error_q3,
        mean_squared_error_avg = EXCLUDED.mean_squared_error_avg,mean_squared_error_stddev = EXCLUDED.mean_squared_error_stddev,
        mean_squared_error_q1 = EXCLUDED.mean_squared_error_q1,mean_squared_error_median = EXCLUDED.mean_squared_error_median,mean_squared_error_q3 = EXCLUDED.mean_squared_error_q3,
        test_kullback_leibler_divergence_avg = EXCLUDED.test_kullback_leibler_divergence_avg,test_kullback_leibler_divergence_stddev = EXCLUDED.test_kullback_leibler_divergence_stddev,
        test_kullback_leibler_divergence_q1 = EXCLUDED.test_kullback_leibler_divergence_q1,test_kullback_leibler_divergence_median = EXCLUDED.test_kullback_leibler_divergence_median,test_kullback_leibler_divergence_q3 = EXCLUDED.test_kullback_leibler_divergence_q3,
        kullback_leibler_divergence_avg = EXCLUDED.kullback_leibler_divergence_avg,kullback_leibler_divergence_stddev = EXCLUDED.kullback_leibler_divergence_stddev,
        kullback_leibler_divergence_q1 = EXCLUDED.kullback_leibler_divergence_q1,kullback_leibler_divergence_median = EXCLUDED.kullback_leibler_divergence_median,kullback_leibler_divergence_q3 = EXCLUDED.kullback_leibler_divergence_q3,
        test_loss_min_epoch_min = EXCLUDED.test_loss_min_epoch_min, test_loss_min_epoch_max = EXCLUDED.test_loss_min_epoch_max, test_loss_min_epoch_avg = EXCLUDED.test_loss_min_epoch_avg, test_loss_min_epoch_q1 = EXCLUDED.test_loss_min_epoch_q1, test_loss_min_epoch_median = EXCLUDED.test_loss_min_epoch_median, test_loss_min_epoch_q3 = EXCLUDED.test_loss_min_epoch_q3, test_loss_min_value_min = EXCLUDED.test_loss_min_value_min, test_loss_min_value_max = EXCLUDED.test_loss_min_value_max, test_loss_min_value_avg = EXCLUDED.test_loss_min_value_avg, test_loss_min_value_q1 = EXCLUDED.test_loss_min_value_q1, test_loss_min_value_median = EXCLUDED.test_loss_min_value_median, test_loss_min_value_q3 = EXCLUDED.test_loss_min_value_q3, test_accuracy_max_epoch_min = EXCLUDED.test_accuracy_max_epoch_min, test_accuracy_max_epoch_max = EXCLUDED.test_accuracy_max_epoch_max, test_accuracy_max_epoch_avg = EXCLUDED.test_accuracy_max_epoch_avg, test_accuracy_max_epoch_q1 = EXCLUDED.test_accuracy_max_epoch_q1, test_accuracy_max_epoch_median = EXCLUDED.test_accuracy_max_epoch_median, test_accuracy_max_epoch_q3 = EXCLUDED.test_accuracy_max_epoch_q3, test_accuracy_max_value_min = EXCLUDED.test_accuracy_max_value_min, test_accuracy_max_value_max = EXCLUDED.test_accuracy_max_value_max, test_accuracy_max_value_avg = EXCLUDED.test_accuracy_max_value_avg, test_accuracy_max_value_q1 = EXCLUDED.test_accuracy_max_value_q1, test_accuracy_max_value_median = EXCLUDED.test_accuracy_max_value_median, test_accuracy_max_value_q3 = EXCLUDED.test_accuracy_max_value_q3, test_mean_squared_error_min_epoch_min = EXCLUDED.test_mean_squared_error_min_epoch_min, test_mean_squared_error_min_epoch_max = EXCLUDED.test_mean_squared_error_min_epoch_max, test_mean_squared_error_min_epoch_avg = EXCLUDED.test_mean_squared_error_min_epoch_avg, test_mean_squared_error_min_epoch_q1 = EXCLUDED.test_mean_squared_error_min_epoch_q1, test_mean_squared_error_min_epoch_median = EXCLUDED.test_mean_squared_error_min_epoch_median, test_mean_squared_error_min_epoch_q3 = EXCLUDED.test_mean_squared_error_min_epoch_q3, test_mean_squared_error_min_value_min = EXCLUDED.test_mean_squared_error_min_value_min, test_mean_squared_error_min_value_max = EXCLUDED.test_mean_squared_error_min_value_max, test_mean_squared_error_min_value_avg = EXCLUDED.test_mean_squared_error_min_value_avg, test_mean_squared_error_min_value_q1 = EXCLUDED.test_mean_squared_error_min_value_q1, test_mean_squared_error_min_value_median = EXCLUDED.test_mean_squared_error_min_value_median, test_mean_squared_error_min_value_q3 = EXCLUDED.test_mean_squared_error_min_value_q3, test_kullback_leibler_divergence_min_epoch_min = EXCLUDED.test_kullback_leibler_divergence_min_epoch_min, test_kullback_leibler_divergence_min_epoch_max = EXCLUDED.test_kullback_leibler_divergence_min_epoch_max, test_kullback_leibler_divergence_min_epoch_avg = EXCLUDED.test_kullback_leibler_divergence_min_epoch_avg, test_kullback_leibler_divergence_min_epoch_q1 = EXCLUDED.test_kullback_leibler_divergence_min_epoch_q1, test_kullback_leibler_divergence_min_epoch_median = EXCLUDED.test_kullback_leibler_divergence_min_epoch_median, test_kullback_leibler_divergence_min_epoch_q3 = EXCLUDED.test_kullback_leibler_divergence_min_epoch_q3, test_kullback_leibler_divergence_min_value_min = EXCLUDED.test_kullback_leibler_divergence_min_value_min, test_kullback_leibler_divergence_min_value_max = EXCLUDED.test_kullback_leibler_divergence_min_value_max, test_kullback_leibler_divergence_min_value_avg = EXCLUDED.test_kullback_leibler_divergence_min_value_avg, test_kullback_leibler_divergence_min_value_q1 = EXCLUDED.test_kullback_leibler_divergence_min_value_q1, test_kullback_leibler_divergence_min_value_median = EXCLUDED.test_kullback_leibler_divergence_min_value_median, test_kullback_leibler_divergence_min_value_q3 = EXCLUDED.test_kullback_leibler_divergence_min_value_q3
            ;"""))
            rows_affected = cursor.rowcount
            if rows_affected == 0:
                break
            print(f'Updated {rows_affected} summary entries.')
    print("Done.")


if __name__ == "__main__":
    main()
