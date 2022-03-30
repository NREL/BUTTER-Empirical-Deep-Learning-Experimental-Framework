SELECT *
FROM pg_stat_activity 
WHERE usename = 'dmpappsops'
ORDER BY state, query_start desc;


SELECT
   relname  as table_name,
   pg_size_pretty(pg_total_relation_size(relid)) As "Total Size",
   pg_size_pretty(pg_relation_size(relid)) as "Core",
   pg_size_pretty(pg_indexes_size(relid)) as "Index",
   pg_size_pretty(pg_table_size(relid) - pg_relation_size(relid) - pg_relation_size(relid, 'vm') - pg_relation_size(relid, 'fsm')) as "TOAST",
   pg_size_pretty(pg_relation_size(relid, 'vm')) as "Visibility Map",
   pg_size_pretty(pg_relation_size(relid, 'fsm')) as "Free Space Map",
   (pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid)) as "Tuples",
   pg_stat_get_live_tuples(relid) as "Live Tuples",
   pg_stat_get_dead_tuples(relid) as "Dead Tuples",
   pg_size_pretty(pg_total_relation_size(relid) / (1 + pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid))) as "Per Tuple",
   pg_size_pretty(pg_relation_size(relid) / (1 + pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid))) as "Core Per Tuple",
   pg_size_pretty(pg_indexes_size(relid) / (1 + pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid))) as "Index Per Tuple",
   pg_size_pretty((pg_table_size(relid) - pg_relation_size(relid) - pg_relation_size(relid, 'vm') - pg_relation_size(relid, 'fsm')) / (1 + pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid))) as "TOAST Per Tuple"
   FROM pg_catalog.pg_statio_user_tables 
ORDER BY pg_total_relation_size(relid) DESC;


set random_page_cost = 1.01 ;
-- set max_worker_processes = 16;
set max_parallel_workers = 8;
set max_parallel_workers_per_gather = 8;
set parallel_setup_cost = 0;
set parallel_tuple_cost = 0.01;
set enable_partitionwise_join to on;
set enable_partitionwise_aggregate to on;

-- set join_collapse_limit = 1;
-- set from_collapse_limit = 1;
set force_parallel_mode to on;

SET enable_bitmapscan TO on;
set plan_cache_mode TO force_custom_plan;


CREATE TABLE IF NOT EXISTS run
(
    experiment_id integer NOT NULL,
    batch smallint not null,
    dataset smallint not null,
    task smallint not null,
    optimizer smallint not null,
    activation smallint not null,
    topology smallint not null,
    learning_rate smallint not null,
    label_noise smallint not null,
    batch_size smallint not null,
    validation_split smallint not null,
    budget smallint not null,
    depth smallint not null,
    val_loss real[] not null,
    loss real[] not null,
    record_timestamp integer not null,
    val_accuracy real[],
    accuracy real[],
    val_mean_squared_error real[],
    mean_squared_error real[],
    val_mean_absolute_error real[],
    mean_absolute_error real[],
    val_root_mean_squared_error real[],
    root_mean_squared_error real[],
    val_mean_squared_logarithmic_error real[],
    mean_squared_logarithmic_error real[],
    val_hinge real[],
    hinge real[],
    val_squared_hinge real[],
    squared_hinge real[],
    val_cosine_similarity real[],
    cosine_similarity real[],
    val_kullback_leibler_divergence real[],
    kullback_leibler_divergence real[],
    log_id uuid NOT NULL
)

alter table run set (autovacuum_enabled = 0);
alter table run set (fillfactor = 100);
alter table run set (parallel_workers = 8);

alter table run alter column val_loss set storage external;
alter table run alter column loss set storage external;
alter table run alter column val_accuracy set storage external;
alter table run alter column accuracy set storage external;
alter table run alter column val_mean_squared_error set storage external;
alter table run alter column mean_squared_error set storage external;
alter table run alter column val_mean_absolute_error set storage external;
alter table run alter column mean_absolute_error set storage external;
alter table run alter column val_root_mean_squared_error set storage external;
alter table run alter column root_mean_squared_error set storage external;
alter table run alter column val_mean_squared_logarithmic_error set storage external;
alter table run alter column mean_squared_logarithmic_error set storage external;
alter table run alter column val_hinge set storage external;
alter table run alter column hinge set storage external;
alter table run alter column val_squared_hinge set storage external;
alter table run alter column squared_hinge set storage external;
alter table run alter column val_cosine_similarity set storage external;
alter table run alter column cosine_similarity set storage external;
alter table run alter column val_kullback_leibler_divergence set storage external;
alter table run alter column kullback_leibler_divergence set storage external;

insert into run select
    r.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    val_loss ,
    loss ,
    0,
    val_accuracy ,
    accuracy ,
    val_mean_squared_error ,
    mean_squared_error ,
    val_mean_absolute_error ,
    mean_absolute_error ,
    val_root_mean_squared_error ,
    root_mean_squared_error ,
    val_mean_squared_logarithmic_error ,
    mean_squared_logarithmic_error ,
    val_hinge ,
    hinge ,
    val_squared_hinge ,
    squared_hinge ,
    val_cosine_similarity ,
    cosine_similarity ,
    val_kullback_leibler_divergence ,
    kullback_leibler_divergence,
    log_id
from 
  run2 r,
  experiment e
where
  r.experiment_id = e.experiment_id
order by
  batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth;

create index run3_gin on run using GIN ((array[batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth]))
create index run3_brin on run using brin (experiment_id, batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth);
CREATE INDEX run3_experiment_id on run (experiment_id);
create index run3_record_timestamp on run (record_timestamp, experiment_id);

--- CREATE INDEX run3_properties on run (task, activation, batch_size, optimizer, dataset, validation_split, learning_rate, label_noise, budget, depth, batch, topology, experiment_id);
--- CREATE INDEX run3_properties2 on run (experiment_id, task, activation, batch_size, optimizer, dataset, validation_split, learning_rate, label_noise, budget, depth, batch, topology);
--- create index run3_brin on run using brin (experiment_id, batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth);

vacuum analyze run;
alter table run set (autovacuum_enabled = 1);


create table experiment_aggregation_val_loss (
    experiment_id integer,
    batch smallint not null,
    dataset smallint not null,
    task smallint not null,
    optimizer smallint not null,
    activation smallint not null,
    topology smallint not null,
    learning_rate smallint not null,
    label_noise smallint not null,
    batch_size smallint not null,
    validation_split smallint not null,
    budget smallint not null,
    depth smallint not null,
    num smallint not null,
    update_timestamp integer not null,
    val_loss_stddev real[] not null,
    val_loss_min real[] not null,
    val_loss_max real[] not null,
    val_loss_percentile real[] not null
);


create table experiment_aggregation_val_loss (
    experiment_id integer not null,
    batch smallint not null,
    dataset smallint not null,
    task smallint not null,
    optimizer smallint not null,
    activation smallint not null,
    topology smallint not null,
    learning_rate smallint not null,
    label_noise smallint not null,
    batch_size smallint not null,
    validation_split smallint not null,
    budget smallint not null,
    depth smallint not null,
    num_runs smallint not null,
    update_timestamp integer not null,
    num smallint[] not null,
    val_loss_avg real[] not null,
    val_loss_stddev real[] not null,
    val_loss_min real[] not null,
    val_loss_max real[] not null,
    val_loss_percentile real[] not null
);

insert into experiment_aggregation_val_loss
(
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,
    val_loss_avg,
    val_loss_stddev,
    val_loss_min,
    val_loss_max,
    val_loss_percentile
from
    (
        select
            *
        from experiment
        where 
            experiment_id > 8028 * 13 and
            experiment_id <= 8028 * 14
    ) s,
    lateral 
    (
        select
            max(num) num_runs,
            array_agg(num) num,
            array_agg(val_loss_avg) val_loss_avg,
            array_agg(val_loss_stddev) val_loss_stddev,
            array_agg(val_loss_min) val_loss_min,
            array_agg(val_loss_max) val_loss_max,
            array_agg(val_loss_percentile) val_loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real val_loss_avg,
                (stddev_samp(v))::real val_loss_stddev,
                MIN(v) val_loss_min,
                MAX(v) val_loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) val_loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    run r,
                    unnest(r.val_loss) WITH ORDINALITY as epoch_value(v, epoch)
                where v is not null and r.experiment_id = s.experiment_id
                order by epoch, v
            ) e
            group by epoch
            order by epoch
        ) x
    ) x 
);



alter table experiment_aggregation_val_loss add primary key (experiment_id);
create index on experiment_aggregation_val_loss (update_timestamp, experiment_id);
create index experiment_aggregation_val_loss_gin on experiment_aggregation_val_loss using GIN ((array[batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth]));
create index experiment_aggregation_val_loss_brin on experiment_aggregation_val_loss using brin (experiment_id, batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth);
vacuum analyze experiment_aggregation_val_loss;


insert into experiment_aggregation_val_loss
(
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,
    val_loss_avg,
    val_loss_stddev,
    val_loss_min,
    val_loss_max,
    val_loss_percentile
from
    (
        select distinct r.experiment_id 
        from run r
        where 
--                     r.experiment_id = agg.experiment_id and 
--                     r.record_timestamp >= agg.update_timestamp and
            r.record_timestamp >= (select max(update_timestamp) from experiment_aggregation_val_loss)
    ) s inner join run r on (r.experiment_id = s.experiment_id),
    lateral 
    (
        select
            max(num) num_runs,
            array_agg(num) num,
            array_agg(val_loss_avg) val_loss_avg,
            array_agg(val_loss_stddev) val_loss_stddev,
            array_agg(val_loss_min) val_loss_min,
            array_agg(val_loss_max) val_loss_max,
            array_agg(val_loss_percentile) val_loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real val_loss_avg,
                (stddev_samp(v))::real val_loss_stddev,
                MIN(v) val_loss_min,
                MAX(v) val_loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) val_loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    unnest(r.val_loss) WITH ORDINALITY as epoch_value(v, epoch)
                where v is not null
                order by epoch, v
            ) e
            group by epoch
            order by epoch
        ) x
    ) x 
)
on conflict (experiment_id) do update set
    update_timestamp = EXCLUDED.update_timestamp,
    num = EXCLUDED.num,
    val_loss_avg = EXCLUDED.val_loss_avg,
    val_loss_stddev = EXCLUDED.val_loss_stddev,
    val_loss_min = EXCLUDED.val_loss_min,
    val_loss_max = EXCLUDED.val_loss_max,
    val_loss_percentile = EXCLUDED.val_loss_percentile
;

-------------------

create table experiment_aggregation_loss (
    experiment_id integer not null,
    batch smallint not null,
    dataset smallint not null,
    task smallint not null,
    optimizer smallint not null,
    activation smallint not null,
    topology smallint not null,
    learning_rate smallint not null,
    label_noise smallint not null,
    batch_size smallint not null,
    validation_split smallint not null,
    budget smallint not null,
    depth smallint not null,
    num_runs smallint not null,
    update_timestamp integer not null,
    num smallint[] not null,
    loss_avg real[] not null,
    loss_stddev real[] not null,
    loss_min real[] not null,
    loss_max real[] not null,
    loss_percentile real[] not null
);

insert into experiment_aggregation_loss
(
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,
    loss_avg,
    loss_stddev,
    loss_min,
    loss_max,
    loss_percentile
from
    (
        select
            *
        from experiment
        where 
            experiment_id > 8028 * 13 and
            experiment_id <= 8028 * 14
    ) s,
    lateral 
    (
        select
            max(num) num_runs,
            array_agg(num) num,
            array_agg(loss_avg) loss_avg,
            array_agg(loss_stddev) loss_stddev,
            array_agg(loss_min) loss_min,
            array_agg(loss_max) loss_max,
            array_agg(loss_percentile) loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real loss_avg,
                (stddev_samp(v))::real loss_stddev,
                MIN(v) loss_min,
                MAX(v) loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    run r,
                    unnest(r.loss) WITH ORDINALITY as epoch_value(v, epoch)
                where v is not null and r.experiment_id = s.experiment_id
                order by epoch, v
            ) e
            group by epoch
            order by epoch
        ) x
    ) x 
);


alter table experiment_aggregation_loss add primary key (experiment_id);
create index on experiment_aggregation_loss (update_timestamp, experiment_id);
create index experiment_aggregation_loss_gin on experiment_aggregation_loss using GIN ((array[batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth]));
create index experiment_aggregation_loss_brin on experiment_aggregation_loss using brin (experiment_id, batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth);
vacuum analyze experiment_aggregation_loss;


insert into experiment_aggregation_loss
(
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,
    loss_avg,
    loss_stddev,
    loss_min,
    loss_max,
    loss_percentile
from
    (
        select distinct r.experiment_id 
        from run r
        where 
--                     r.experiment_id = agg.experiment_id and 
--                     r.record_timestamp >= agg.update_timestamp and
            r.record_timestamp >= (select max(update_timestamp) from experiment_aggregation_loss)
    ) s inner join run r on (r.experiment_id = s.experiment_id),
    lateral 
    (
        select
            max(num) num_runs,
            array_agg(num) num,
            array_agg(loss_avg) loss_avg,
            array_agg(loss_stddev) loss_stddev,
            array_agg(loss_min) loss_min,
            array_agg(loss_max) loss_max,
            array_agg(loss_percentile) loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real loss_avg,
                (stddev_samp(v))::real loss_stddev,
                MIN(v) loss_min,
                MAX(v) loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    unnest(r.loss) WITH ORDINALITY as epoch_value(v, epoch)
                where v is not null
                order by epoch, v
            ) e
            group by epoch
            order by epoch
        ) x
    ) x 
)
on conflict (experiment_id) do update set
    update_timestamp = EXCLUDED.update_timestamp,
    num = EXCLUDED.num,
    loss_avg = EXCLUDED.loss_avg,
    loss_stddev = EXCLUDED.loss_stddev,
    loss_min = EXCLUDED.loss_min,
    loss_max = EXCLUDED.loss_max,
    loss_percentile = EXCLUDED.loss_percentile
;

-------------------


--- materialized aggregate form: 80ms

select 
    *
from
    experiment_aggregation_val_loss
where
    batch = 15 and learning_rate = 48 and label_noise = 51;
    

-- rejoin form: 13:47
    
select
    x.experiment_id,
    batch, 
    dataset,
    task,
    optimizer,
    activation,
    topology,
    learning_rate,
    label_noise,
    batch_size,
    validation_split,
    budget,
    depth,
    num_runs,
    1::integer update_timestamp,
    num,
    val_loss_avg,
    val_loss_stddev,
    val_loss_min,
    val_loss_max,
    val_loss_percentile
from
(
    select
        experiment_id,
        max(num) num_runs,
        array_agg(num) num,
        array_agg(val_loss_avg) val_loss_avg,
        array_agg(val_loss_stddev) val_loss_stddev,
        array_agg(val_loss_min) val_loss_min,
        array_agg(val_loss_max) val_loss_max,
        array_agg(val_loss_percentile) val_loss_percentile
    from (
    select 
        experiment_id,
        (COUNT(*))::smallint num,
        (AVG(v))::real val_loss_avg,
        (stddev_samp(v))::real val_loss_stddev,
        MIN(v) val_loss_min,
        MAX(v) val_loss_max,
        (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) val_loss_percentile
    from (
        select
            experiment_id, epoch::smallint epoch, v
        from
            run r,
            lateral unnest(r.val_loss) WITH ORDINALITY as epoch_value(v, epoch)
        where
            batch = 15 and learning_rate = 48 and label_noise = 51
            and v is not null
        order by experiment_id, epoch, v
    ) e
    group by experiment_id, epoch
    order by experiment_id, epoch
    ) x
    group by experiment_id
) x inner join experiment y on (y.experiment_id = x.experiment_id)
;

--- lateral form: 12:29

select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,
    val_loss_avg,
    val_loss_stddev,
    val_loss_min,
    val_loss_max,
    val_loss_percentile
from
    experiment s,
    lateral 
    (
        select
            max(num) num_runs,
            array_agg(num) num,
            array_agg(val_loss_avg) val_loss_avg,
            array_agg(val_loss_stddev) val_loss_stddev,
            array_agg(val_loss_min) val_loss_min,
            array_agg(val_loss_max) val_loss_max,
            array_agg(val_loss_percentile) val_loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real val_loss_avg,
                (stddev_samp(v))::real val_loss_stddev,
                MIN(v) val_loss_min,
                MAX(v) val_loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) val_loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    run r,
                    unnest(r.val_loss) WITH ORDINALITY as epoch_value(v, epoch)
                where v is not null and r.experiment_id = s.experiment_id
                order by epoch, v
            ) e
            group by epoch
            order by epoch
        ) x
    ) x 
where batch = 15 and learning_rate = 48 and label_noise = 51
;




-------------------


create table experiment_aggregation (
    experiment_id integer not null,
    batch smallint not null,
    dataset smallint not null,
    task smallint not null,
    optimizer smallint not null,
    activation smallint not null,
    topology smallint not null,
    learning_rate smallint not null,
    label_noise smallint not null,
    batch_size smallint not null,
    validation_split smallint not null,
    budget smallint not null,
    depth smallint not null,
    num_runs smallint not null,
    update_timestamp integer not null,
    
    num smallint[] not null,
    
    val_loss_num_finite smallint[],
    val_loss_avg real[],
    val_loss_stddev real[],
    val_loss_min real[],
    val_loss_max real[],
    val_loss_percentile real[],
    
    loss_num_finite smallint[],
    loss_avg real[],
    loss_stddev real[],
    loss_min real[],
    loss_max real[],
    loss_percentile real[],

    val_accuracy_avg real[],
    val_accuracy_stddev real[],
    accuracy_avg real[],
    accuracy_stddev real[],
    val_mean_squared_error_avg real[],
    val_mean_squared_error_stddev real[],
    mean_squared_error_avg real[],
    mean_squared_error_stddev real[],
    
    val_kullback_leibler_divergence_avg real[],
    val_kullback_leibler_divergence_stddev real[],
    kullback_leibler_divergence_avg real[],
    kullback_leibler_divergence_stddev real[]
);




insert into experiment_aggregation
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,

    val_loss_num_finite,
    val_loss_avg,
    val_loss_stddev,
    val_loss_min,
    val_loss_max,
    val_loss_percentile,

    loss_num_finite,
    loss_avg,
    loss_stddev,
    loss_min,
    loss_max,
    loss_percentile,

    val_accuracy_avg,
    val_accuracy_stddev,
    accuracy_avg,
    accuracy_stddev,
    val_mean_squared_error_avg,
    val_mean_squared_error_stddev,
    mean_squared_error_avg,
    mean_squared_error_stddev,
    
    val_kullback_leibler_divergence_avg,
    val_kullback_leibler_divergence_stddev,
    kullback_leibler_divergence_avg,
    kullback_leibler_divergence_stddev
from
   (
        select
            *
        from experiment
        where 
            experiment_id >= 8028 * 0 and
            experiment_id < 8028 * 2
    ) s,  
    lateral 
    (
        select
            max(num) num_runs,
            array_agg(num) num,

            array_agg(val_loss_num_finite) val_loss_num_finite,
            array_agg(val_loss_avg) val_loss_avg,
            array_agg(val_loss_stddev) val_loss_stddev,
            array_agg(val_loss_min) val_loss_min,
            array_agg(val_loss_max) val_loss_max,
            array_agg(val_loss_percentile) val_loss_percentile,
        
            array_agg(loss_num_finite) loss_num_finite,
            array_agg(loss_avg) loss_avg,
            array_agg(loss_stddev) loss_stddev,
            array_agg(loss_min) loss_min,
            array_agg(loss_max) loss_max,
            array_agg(loss_percentile) loss_percentile,
                   
            array_agg(val_accuracy_avg) val_accuracy_avg,
            array_agg(val_accuracy_stddev) val_accuracy_stddev,
            array_agg(accuracy_avg) accuracy_avg,
            array_agg(accuracy_stddev) accuracy_stddev,
            array_agg(val_mean_squared_error_avg) val_mean_squared_error_avg,
            array_agg(val_mean_squared_error_stddev) val_mean_squared_error_stddev,
            array_agg(mean_squared_error_avg) mean_squared_error_avg,
            array_agg(mean_squared_error_stddev) mean_squared_error_stddev,
            
            array_agg(val_kullback_leibler_divergence_avg) val_kullback_leibler_divergence_avg,
            array_agg(val_kullback_leibler_divergence_stddev) val_kullback_leibler_divergence_stddev,
            array_agg(kullback_leibler_divergence_avg) kullback_leibler_divergence_avg,
            array_agg(kullback_leibler_divergence_stddev) kullback_leibler_divergence_stddev
        from (
            select 
                (COUNT(val_loss_nan))::smallint num,
            
                (COUNT(val_loss))::smallint val_loss_num_finite,
                (AVG(val_loss))::real val_loss_avg,
                (stddev_samp(val_loss))::real val_loss_stddev,
                MIN(val_loss) val_loss_min,
                MAX(val_loss) val_loss_max,
                (PERCENTILE_DISC(array[.166, .333, .5, .667, .834]) WITHIN GROUP(ORDER BY val_loss_nan)) val_loss_percentile,
            
                (COUNT(loss))::smallint loss_num_finite,
                (AVG(loss))::real loss_avg,
                (stddev_samp(loss))::real loss_stddev,
                MIN(loss) loss_min,
                MAX(loss) loss_max,
                (PERCENTILE_DISC(array[.166, .333, .5, .667, .834]) WITHIN GROUP(ORDER BY loss_nan)) loss_percentile,
            
                (AVG(val_accuracy))::real val_accuracy_avg,
                (stddev_samp(val_accuracy))::real val_accuracy_stddev,
                (AVG(accuracy))::real accuracy_avg,
                (stddev_samp(accuracy))::real accuracy_stddev,
                (AVG(val_mean_squared_error))::real val_mean_squared_error_avg,
                (stddev_samp(val_mean_squared_error))::real val_mean_squared_error_stddev,
                (AVG(mean_squared_error))::real mean_squared_error_avg,
                (stddev_samp(mean_squared_error))::real mean_squared_error_stddev,
                
                (AVG(val_kullback_leibler_divergence))::real val_kullback_leibler_divergence_avg,
                (stddev_samp(val_kullback_leibler_divergence))::real val_kullback_leibler_divergence_stddev,
                (AVG(kullback_leibler_divergence))::real kullback_leibler_divergence_avg,
                (stddev_samp(kullback_leibler_divergence))::real kullback_leibler_divergence_stddev
            from (
                select
                    epoch::smallint epoch, 
                    v val_loss,
                    COALESCE(v, 'NaN'::real) val_loss_nan,
                    loss[epoch] loss,
                    COALESCE(loss[epoch], 'NaN'::real) loss_nan,
                
                    val_accuracy[epoch] val_accuracy,
                    accuracy[epoch] accuracy,
                    val_mean_squared_error[epoch] val_mean_squared_error,
                    mean_squared_error[epoch] mean_squared_error,
                    
                    val_kullback_leibler_divergence[epoch] val_kullback_leibler_divergence,
                    kullback_leibler_divergence[epoch] kullback_leibler_divergence
                from
                    run r,
                    unnest(r.val_loss) WITH ORDINALITY as epoch_value(v, epoch)
                where v is not null and r.experiment_id = s.experiment_id
                order by epoch, v
            ) e
            group by epoch
            order by epoch
        ) x
    ) x 
;



select experiment_id, ep.experiment_parameters
from
(select experiment_id, experiment_parameters from experiment_ limit 4) e,
lateral (
    select array_agg(pid::smallint) experiment_parameters from
    (
        select pid
        from
        lateral unnest(e.experiment_parameters) pid
        union all
        select pid from (values (24), (27)) as t(pid)
        order by pid
    ) pid
    ) ep
;

update experiment_ e
set experiment_parameters= 
(
    select array_agg(pid::smallint) experiment_parameters from
    (
        select pid
        from
        lateral unnest(e.experiment_parameters) pid
        union all
        select pid from (values (24), (27)) as t(pid)
        order by pid
    ) pid
);

update run_ r
set run_parameters = 
(
    select array_agg(pid::smallint) run_parameters from
    (
        select pid
        from
        lateral unnest(r.run_parameters) pid
        union all
        select pid from (values (24), (27)) as t(pid)
        order by pid
    ) pid
);


select * from parameter_ order by kind, real_value, integer_value, string_value;

with param as (
select * from parameter_ p 
)
select param.*, ec.* from 
param,
lateral (
    select count(*) num 
    from experiment_ e
    where e.experiment_parameters @> (ARRAY[param.id])::smallint[]
    ) ec
order by param.kind, ec.num, param.real_value, param.integer_value, param.string_value;


