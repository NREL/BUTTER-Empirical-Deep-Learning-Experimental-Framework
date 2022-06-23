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
    val_loss_num_finite, val_loss_min, val_loss_max,
    val_loss_avg, val_loss_stddev,    
    val_loss_q1, val_loss_median, val_loss_q3,
    loss_num_finite, loss_min, loss_max,
    loss_avg, loss_stddev,
    loss_q1, loss_median, loss_q3,
    val_accuracy_avg, val_accuracy_stddev,
    val_accuracy_q1, val_accuracy_median, val_accuracy_q3,
    accuracy_avg, accuracy_stddev,
    accuracy_q1, accuracy_median, accuracy_q3,
    val_mean_squared_error_avg, val_mean_squared_error_stddev,
    val_mean_squared_error_q1, val_mean_squared_error_median, val_mean_squared_error_q3,
    mean_squared_error_avg, mean_squared_error_stddev,
    mean_squared_error_q1, mean_squared_error_median, mean_squared_error_q3,
    val_kullback_leibler_divergence_avg, val_kullback_leibler_divergence_stddev,
    val_kullback_leibler_divergence_q1, val_kullback_leibler_divergence_median, val_kullback_leibler_divergence_q3,
    kullback_leibler_divergence_avg, kullback_leibler_divergence_stddev,
    kullback_leibler_divergence_q1, kullback_leibler_divergence_median, kullback_leibler_divergence_q3
)
select
    e.experiment_id,
    experiment_parameters,
    widths,
    network_structure,
    e.size,
    e.num_free_parameters,
    e.relative_size_error,
    aggregates.*
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
--  order by e.experiment_id asc
    limit 16
    for update
    skip locked
--    order by experiment_id offset 0 limit 4
) e,
lateral (
    with rr as (
        select run_.ctid as run_ctid from run_ where run_.experiment_id = e.experiment_id
    )
    select * from
        (
            select
                max(v_num) num_runs, array_agg(v_num) num,
                array_agg(v_num_finite) val_loss_num_finite, array_agg(v_min) val_loss_min, array_agg(v_max) val_loss_max,    
                array_agg(v_avg) val_loss_avg, array_agg(v_stddev) val_loss_stddev,                
                array_agg(v_percentile[1]) val_loss_q1, array_agg(v_percentile[2]) val_loss_median, array_agg(v_percentile[3]) val_loss_q3
            from
            (       
                select
                    (COUNT(coalesce(v, 'NaN'::real)))::smallint v_num, 
                    (COUNT(v))::smallint v_num_finite, MIN(v) v_min, MAX(v) v_max,    
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev,
                    (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real))) v_percentile
                from
                    rr inner join run_ r on rr.run_ctid = r.ctid,
                    unnest(r.val_loss) WITH ORDINALITY as epoch_value(v, epoch)
                group by epoch order by epoch
            ) x
        ) val_loss_,
        lateral (
            select
                array_agg(v_num_finite) loss_num_finite, array_agg(v_min) loss_min, array_agg(v_max) loss_max,
                array_agg(v_avg) loss_avg, array_agg(v_stddev) loss_stddev,
                array_agg(v_percentile[1]) loss_q1, array_agg(v_percentile[2]) loss_median, array_agg(v_percentile[3]) loss_q3
            from
            (       
                select
                    (COUNT(v))::smallint v_num_finite, MIN(v) v_min, MAX(v) v_max,    
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev,
                    (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real))) v_percentile
                from
                    rr inner join run_ r on rr.run_ctid = r.ctid,
                    unnest(r.loss) WITH ORDINALITY as epoch_value(v, epoch)
                group by epoch order by epoch
            ) x
        ) loss_,
        lateral (select array_agg(v_avg) val_accuracy_avg, array_agg(v_stddev) val_accuracy_stddev, array_agg(v_percentile[1]) val_accuracy_q1, array_agg(v_percentile[2]) val_accuracy_median, array_agg(v_percentile[3]) val_accuracy_q3 from
                (select
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real))) v_percentile
                    from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.val_accuracy) WITH ORDINALITY as epoch_value(v, epoch)
                    group by epoch order by epoch
                ) x ) val_accuracy_,
        lateral (select array_agg(v_avg) accuracy_avg, array_agg(v_stddev) accuracy_stddev, array_agg(v_percentile[1]) accuracy_q1, array_agg(v_percentile[2]) accuracy_median, array_agg(v_percentile[3]) accuracy_q3 from
                (select
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real))) v_percentile
                    from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.accuracy) WITH ORDINALITY as epoch_value(v, epoch)
                    group by epoch order by epoch
                ) x ) accuracy_,
        lateral (select array_agg(v_avg) val_mean_squared_error_avg, array_agg(v_stddev) val_mean_squared_error_stddev, array_agg(v_percentile[1]) val_mean_squared_error_q1, array_agg(v_percentile[2]) val_mean_squared_error_median, array_agg(v_percentile[3]) val_mean_squared_error_q3 from
                (select
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real))) v_percentile
                    from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.val_mean_squared_error) WITH ORDINALITY as epoch_value(v, epoch)
                    group by epoch order by epoch
                ) x ) val_mean_squared_error_,
        lateral (select array_agg(v_avg) mean_squared_error_avg, array_agg(v_stddev) mean_squared_error_stddev, array_agg(v_percentile[1]) mean_squared_error_q1, array_agg(v_percentile[2]) mean_squared_error_median, array_agg(v_percentile[3]) mean_squared_error_q3 from
                (select
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real))) v_percentile
                    from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.mean_squared_error) WITH ORDINALITY as epoch_value(v, epoch)
                    group by epoch order by epoch
                ) x ) mean_squared_error_,
        lateral (select array_agg(v_avg) val_kullback_leibler_divergence_avg, array_agg(v_stddev) val_kullback_leibler_divergence_stddev, array_agg(v_percentile[1]) val_kullback_leibler_divergence_q1, array_agg(v_percentile[2]) val_kullback_leibler_divergence_median, array_agg(v_percentile[3]) val_kullback_leibler_divergence_q3 from
                (select
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real))) v_percentile
                    from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.val_kullback_leibler_divergence) WITH ORDINALITY as epoch_value(v, epoch)
                    group by epoch order by epoch
                ) x ) val_kullback_leibler_divergence_,
        lateral (select array_agg(v_avg) kullback_leibler_divergence_avg, array_agg(v_stddev) kullback_leibler_divergence_stddev, array_agg(v_percentile[1]) kullback_leibler_divergence_q1, array_agg(v_percentile[2]) kullback_leibler_divergence_median, array_agg(v_percentile[3]) kullback_leibler_divergence_q3 from
                (select
                    (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real))) v_percentile
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
        val_loss_num_finite = EXCLUDED.val_loss_num_finite,val_loss_min = EXCLUDED.val_loss_min,val_loss_max = EXCLUDED.val_loss_max,
        val_loss_avg = EXCLUDED.val_loss_avg,val_loss_stddev = EXCLUDED.val_loss_stddev,    
        val_loss_q1 = EXCLUDED.val_loss_q1,val_loss_median = EXCLUDED.val_loss_median,val_loss_q3 = EXCLUDED.val_loss_q3,
        loss_num_finite = EXCLUDED.loss_num_finite,loss_min = EXCLUDED.loss_min,loss_max = EXCLUDED.loss_max,
        loss_avg = EXCLUDED.loss_avg,loss_stddev = EXCLUDED.loss_stddev,
        loss_q1 = EXCLUDED.loss_q1,loss_median = EXCLUDED.loss_median,loss_q3 = EXCLUDED.loss_q3,
        val_accuracy_avg = EXCLUDED.val_accuracy_avg,val_accuracy_stddev = EXCLUDED.val_accuracy_stddev,
        val_accuracy_q1 = EXCLUDED.val_accuracy_q1,val_accuracy_median = EXCLUDED.val_accuracy_median,val_accuracy_q3 = EXCLUDED.val_accuracy_q3,
        accuracy_avg = EXCLUDED.accuracy_avg,accuracy_stddev = EXCLUDED.accuracy_stddev,
        accuracy_q1 = EXCLUDED.accuracy_q1,accuracy_median = EXCLUDED.accuracy_median,accuracy_q3 = EXCLUDED.accuracy_q3,
        val_mean_squared_error_avg = EXCLUDED.val_mean_squared_error_avg,val_mean_squared_error_stddev = EXCLUDED.val_mean_squared_error_stddev,
        val_mean_squared_error_q1 = EXCLUDED.val_mean_squared_error_q1,val_mean_squared_error_median = EXCLUDED.val_mean_squared_error_median,val_mean_squared_error_q3 = EXCLUDED.val_mean_squared_error_q3,
        mean_squared_error_avg = EXCLUDED.mean_squared_error_avg,mean_squared_error_stddev = EXCLUDED.mean_squared_error_stddev,
        mean_squared_error_q1 = EXCLUDED.mean_squared_error_q1,mean_squared_error_median = EXCLUDED.mean_squared_error_median,mean_squared_error_q3 = EXCLUDED.mean_squared_error_q3,
        val_kullback_leibler_divergence_avg = EXCLUDED.val_kullback_leibler_divergence_avg,val_kullback_leibler_divergence_stddev = EXCLUDED.val_kullback_leibler_divergence_stddev,
        val_kullback_leibler_divergence_q1 = EXCLUDED.val_kullback_leibler_divergence_q1,val_kullback_leibler_divergence_median = EXCLUDED.val_kullback_leibler_divergence_median,val_kullback_leibler_divergence_q3 = EXCLUDED.val_kullback_leibler_divergence_q3,
        kullback_leibler_divergence_avg = EXCLUDED.kullback_leibler_divergence_avg,kullback_leibler_divergence_stddev = EXCLUDED.kullback_leibler_divergence_stddev,
        kullback_leibler_divergence_q1 = EXCLUDED.kullback_leibler_divergence_q1,kullback_leibler_divergence_median = EXCLUDED.kullback_leibler_divergence_median,kullback_leibler_divergence_q3 = EXCLUDED.kullback_leibler_divergence_q3
        ;"""))
            rows_affected = cursor.rowcount
            if rows_affected == 0:
                break
            print(f'Updated {rows_affected} summary entries.')
    print("Done.")

if __name__ == "__main__":
    main()
