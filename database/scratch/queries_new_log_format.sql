SELECT l.locktype, p.pid as pid , p.datname as database, p.usename as user, p.application_name as application, p.query as query,
       b.pid as blocking_pid, b.usename as blocking_user, b.application_name as blocking_application, b.query as blocking_query
  FROM
       pg_locks l,
       pg_stat_activity p,
       pg_locks bl,
       pg_stat_activity b
 WHERE
       p.pid = l.pid AND NOT l.granted AND
       bl.database = l.database AND bl.relation = l.relation AND bl.granted AND
       b.pid = bl.pid;

SELECT *
FROM pg_stat_activity 
WHERE usename = 'dmpappsops'
ORDER BY state, query_start desc;

SELECT *
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name   = 'materialized_experiments_2'
     ;

SELECT * FROM pg_stat_activity WHERE datname = 'dmpapps'  ORDER BY query;

SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'dmpapps' and username = 'dmpappsops';

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
    test_loss real[] not null,
    train_loss real[] not null,
    record_timestamp integer not null,
    test_accuracy real[],
    train_accuracy real[],
    test_mean_squared_error real[],
    train_mean_squared_error real[],
    test_mean_absolute_error real[],
    train_mean_absolute_error real[],
    test_root_mean_squared_error real[],
    train_root_mean_squared_error real[],
    test_mean_squared_logarithmic_error real[],
    train_mean_squared_logarithmic_error real[],
    test_hinge real[],
    train_hinge real[],
    test_squared_hinge real[],
    train_squared_hinge real[],
    test_cosine_similarity real[],
    train_cosine_similarity real[],
    test_kullback_leibler_divergence real[],
    train_kullback_leibler_divergence real[],
    log_id uuid NOT NULL
)

alter table run set (autovacuum_enabled = 0);
alter table run set (fillfactor = 100);
alter table run set (parallel_workers = 8);

alter table run alter column test_loss set storage external;
alter table run alter column train_loss set storage external;
alter table run alter column test_accuracy set storage external;
alter table run alter column train_accuracy set storage external;
alter table run alter column test_mean_squared_error set storage external;
alter table run alter column train_mean_squared_error set storage external;
alter table run alter column test_mean_absolute_error set storage external;
alter table run alter column train_mean_absolute_error set storage external;
alter table run alter column test_root_mean_squared_error set storage external;
alter table run alter column train_root_mean_squared_error set storage external;
alter table run alter column test_mean_squared_logarithmic_error set storage external;
alter table run alter column train_mean_squared_logarithmic_error set storage external;
alter table run alter column test_hinge set storage external;
alter table run alter column train_hinge set storage external;
alter table run alter column test_squared_hinge set storage external;
alter table run alter column train_squared_hinge set storage external;
alter table run alter column test_cosine_similarity set storage external;
alter table run alter column train_cosine_similarity set storage external;
alter table run alter column test_kullback_leibler_divergence set storage external;
alter table run alter column train_kullback_leibler_divergence set storage external;

insert into run select
    r.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    test_loss ,
    train_loss ,
    0,
    test_accuracy ,
    train_accuracy ,
    test_mean_squared_error ,
    train_mean_squared_error ,
    test_mean_absolute_error ,
    train_mean_absolute_error ,
    test_root_mean_squared_error ,
    train_root_mean_squared_error ,
    test_mean_squared_logarithmic_error ,
    train_mean_squared_logarithmic_error ,
    test_hinge ,
    train_hinge ,
    test_squared_hinge ,
    train_squared_hinge ,
    test_cosine_similarity ,
    train_cosine_similarity ,
    test_kullback_leibler_divergence ,
    train_kullback_leibler_divergence,
    log_id
from 
  run r,
  experiment e
where
  r.experiment_id = e.experiment_id
order by
  batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth;

create index run2_gin on run using GIN ((array[batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth]))
create index run2_brin on run using brin (experiment_id, batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth);
CREATE INDEX run2_experiment_id on run (experiment_id);
create index run2_record_timestamp on run (record_timestamp, experiment_id);

--- CREATE INDEX run2_properties on run (task, activation, batch_size, optimizer, dataset, validation_split, learning_rate, label_noise, budget, depth, batch, topology, experiment_id);
--- CREATE INDEX run2_properties2 on run (experiment_id, task, activation, batch_size, optimizer, dataset, validation_split, learning_rate, label_noise, budget, depth, batch, topology);
--- create index run2_brin on run using brin (experiment_id, batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth);

vacuum analyze run;
alter table run set (autovacuum_enabled = 1);


create table experiment_aggregation_test_loss (
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
    test_loss_stddev real[] not null,
    test_loss_min real[] not null,
    test_loss_max real[] not null,
    test_loss_percentile real[] not null
);


create table experiment_aggregation_test_loss (
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
    test_loss_avg real[] not null,
    test_loss_stddev real[] not null,
    test_loss_min real[] not null,
    test_loss_max real[] not null,
    test_loss_percentile real[] not null
);

insert into experiment_aggregation_test_loss
(
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,
    test_loss_avg,
    test_loss_stddev,
    test_loss_min,
    test_loss_max,
    test_loss_percentile
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
            array_agg(test_loss_avg) test_loss_avg,
            array_agg(test_loss_stddev) test_loss_stddev,
            array_agg(test_loss_min) test_loss_min,
            array_agg(test_loss_max) test_loss_max,
            array_agg(test_loss_percentile) test_loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real test_loss_avg,
                (stddev_samp(v))::real test_loss_stddev,
                MIN(v) test_loss_min,
                MAX(v) test_loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) test_loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    run r,
                    unnest(r.test_loss) WITH ORDINALITY as epoch_value(v, epoch)
                where v is not null and r.experiment_id = s.experiment_id
                order by epoch, v
            ) e
            group by epoch
            order by epoch
        ) x
    ) x 
);



alter table experiment_aggregation_test_loss add primary key (experiment_id);
create index on experiment_aggregation_test_loss (update_timestamp, experiment_id);
create index experiment_aggregation_test_loss_gin on experiment_aggregation_test_loss using GIN ((array[batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth]));
create index experiment_aggregation_test_loss_brin on experiment_aggregation_test_loss using brin (experiment_id, batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth);
vacuum analyze experiment_aggregation_test_loss;


insert into experiment_aggregation_test_loss
(
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,
    test_loss_avg,
    test_loss_stddev,
    test_loss_min,
    test_loss_max,
    test_loss_percentile
from
    (
        select distinct r.experiment_id 
        from run r
        where 
--                     r.experiment_id = agg.experiment_id and 
--                     r.record_timestamp >= agg.update_timestamp and
            r.record_timestamp >= (select max(update_timestamp) from experiment_aggregation_test_loss)
    ) s inner join run r on (r.experiment_id = s.experiment_id),
    lateral 
    (
        select
            max(num) num_runs,
            array_agg(num) num,
            array_agg(test_loss_avg) test_loss_avg,
            array_agg(test_loss_stddev) test_loss_stddev,
            array_agg(test_loss_min) test_loss_min,
            array_agg(test_loss_max) test_loss_max,
            array_agg(test_loss_percentile) test_loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real test_loss_avg,
                (stddev_samp(v))::real test_loss_stddev,
                MIN(v) test_loss_min,
                MAX(v) test_loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) test_loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    unnest(r.test_loss) WITH ORDINALITY as epoch_value(v, epoch)
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
    test_loss_avg = EXCLUDED.test_loss_avg,
    test_loss_stddev = EXCLUDED.test_loss_stddev,
    test_loss_min = EXCLUDED.test_loss_min,
    test_loss_max = EXCLUDED.test_loss_max,
    test_loss_percentile = EXCLUDED.test_loss_percentile
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
    train_loss_avg real[] not null,
    train_loss_stddev real[] not null,
    train_loss_min real[] not null,
    train_loss_max real[] not null,
    train_loss_percentile real[] not null
);

insert into experiment_aggregation_loss
(
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,
    train_loss_avg,
    train_loss_stddev,
    train_loss_min,
    train_loss_max,
    train_loss_percentile
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
            array_agg(train_loss_avg) train_loss_avg,
            array_agg(train_loss_stddev) train_loss_stddev,
            array_agg(train_loss_min) train_loss_min,
            array_agg(train_loss_max) train_loss_max,
            array_agg(train_loss_percentile) train_loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real train_loss_avg,
                (stddev_samp(v))::real train_loss_stddev,
                MIN(v) train_loss_min,
                MAX(v) train_loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) train_loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    run r,
                    unnest(r.train_loss) WITH ORDINALITY as epoch_value(v, epoch)
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
    train_loss_avg,
    train_loss_stddev,
    train_loss_min,
    train_loss_max,
    train_loss_percentile
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
            array_agg(train_loss_avg) train_loss_avg,
            array_agg(train_loss_stddev) train_loss_stddev,
            array_agg(train_loss_min) train_loss_min,
            array_agg(train_loss_max) train_loss_max,
            array_agg(train_loss_percentile) train_loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real train_loss_avg,
                (stddev_samp(v))::real train_loss_stddev,
                MIN(v) train_loss_min,
                MAX(v) train_loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) train_loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    unnest(r.train_loss) WITH ORDINALITY as epoch_value(v, epoch)
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
    train_loss_avg = EXCLUDED.train_loss_avg,
    train_loss_stddev = EXCLUDED.train_loss_stddev,
    train_loss_min = EXCLUDED.train_loss_min,
    train_loss_max = EXCLUDED.train_loss_max,
    train_loss_percentile = EXCLUDED.train_loss_percentile
;

-------------------


--- materialized aggregate form: 80ms

select 
    *
from
    experiment_aggregation_test_loss
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
    test_loss_avg,
    test_loss_stddev,
    test_loss_min,
    test_loss_max,
    test_loss_percentile
from
(
    select
        experiment_id,
        max(num) num_runs,
        array_agg(num) num,
        array_agg(test_loss_avg) test_loss_avg,
        array_agg(test_loss_stddev) test_loss_stddev,
        array_agg(test_loss_min) test_loss_min,
        array_agg(test_loss_max) test_loss_max,
        array_agg(test_loss_percentile) test_loss_percentile
    from (
    select 
        experiment_id,
        (COUNT(*))::smallint num,
        (AVG(v))::real test_loss_avg,
        (stddev_samp(v))::real test_loss_stddev,
        MIN(v) test_loss_min,
        MAX(v) test_loss_max,
        (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) test_loss_percentile
    from (
        select
            experiment_id, epoch::smallint epoch, v
        from
            run r,
            lateral unnest(r.test_loss) WITH ORDINALITY as epoch_value(v, epoch)
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
    test_loss_avg,
    test_loss_stddev,
    test_loss_min,
    test_loss_max,
    test_loss_percentile
from
    experiment s,
    lateral 
    (
        select
            max(num) num_runs,
            array_agg(num) num,
            array_agg(test_loss_avg) test_loss_avg,
            array_agg(test_loss_stddev) test_loss_stddev,
            array_agg(test_loss_min) test_loss_min,
            array_agg(test_loss_max) test_loss_max,
            array_agg(test_loss_percentile) test_loss_percentile
        from (
            select 
                (COUNT(*))::smallint num,
                (AVG(v))::real test_loss_avg,
                (stddev_samp(v))::real test_loss_stddev,
                MIN(v) test_loss_min,
                MAX(v) test_loss_max,
                (PERCENTILE_DISC(array[.1, .2, .3, .4, .5, .6, .7, .8, .9]) WITHIN GROUP(ORDER BY v)) test_loss_percentile
            from (
                select
                    epoch::smallint epoch, v
                from
                    run r,
                    unnest(r.test_loss) WITH ORDINALITY as epoch_value(v, epoch)
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
    
    test_loss_num_finite smallint[],
    test_loss_avg real[],
    test_loss_stddev real[],
    test_loss_min real[],
    test_loss_max real[],
    test_loss_percentile real[],
    
    train_loss_num_finite smallint[],
    train_loss_avg real[],
    train_loss_stddev real[],
    train_loss_min real[],
    train_loss_max real[],
    train_loss_percentile real[],

    test_accuracy_avg real[],
    test_accuracy_stddev real[],
    train_accuracy_avg real[],
    train_accuracy_stddev real[],
    test_mean_squared_error_avg real[],
    test_mean_squared_error_stddev real[],
    train_mean_squared_error_avg real[],
    train_mean_squared_error_stddev real[],
    
    test_kullback_leibler_divergence_avg real[],
    test_kullback_leibler_divergence_stddev real[],
    train_kullback_leibler_divergence_avg real[],
    train_kullback_leibler_divergence_stddev real[]
);




insert into experiment_aggregation
select
    s.experiment_id,
    batch, dataset, task, optimizer, activation, topology, learning_rate, label_noise, batch_size, validation_split, budget, depth,
    num_runs,
    cast(extract(epoch from current_timestamp) as integer) update_timestamp,
    num,

    test_loss_num_finite,
    test_loss_avg,
    test_loss_stddev,
    test_loss_min,
    test_loss_max,
    test_loss_percentile,

    train_loss_num_finite,
    train_loss_avg,
    train_loss_stddev,
    train_loss_min,
    train_loss_max,
    train_loss_percentile,

    test_accuracy_avg,
    test_accuracy_stddev,
    train_accuracy_avg,
    train_accuracy_stddev,
    test_mean_squared_error_avg,
    test_mean_squared_error_stddev,
    train_mean_squared_error_avg,
    train_mean_squared_error_stddev,
    
    test_kullback_leibler_divergence_avg,
    test_kullback_leibler_divergence_stddev,
    train_kullback_leibler_divergence_avg,
    train_kullback_leibler_divergence_stddev
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

            array_agg(test_loss_num_finite) test_loss_num_finite,
            array_agg(test_loss_avg) test_loss_avg,
            array_agg(test_loss_stddev) test_loss_stddev,
            array_agg(test_loss_min) test_loss_min,
            array_agg(test_loss_max) test_loss_max,
            array_agg(test_loss_percentile) test_loss_percentile,
        
            array_agg(train_loss_num_finite) train_loss_num_finite,
            array_agg(train_loss_avg) train_loss_avg,
            array_agg(train_loss_stddev) train_loss_stddev,
            array_agg(train_loss_min) train_loss_min,
            array_agg(train_loss_max) train_loss_max,
            array_agg(train_loss_percentile) train_loss_percentile,
                   
            array_agg(test_accuracy_avg) test_accuracy_avg,
            array_agg(test_accuracy_stddev) test_accuracy_stddev,
            array_agg(train_accuracy_avg) train_accuracy_avg,
            array_agg(train_accuracy_stddev) train_accuracy_stddev,
            array_agg(test_mean_squared_error_avg) test_mean_squared_error_avg,
            array_agg(test_mean_squared_error_stddev) test_mean_squared_error_stddev,
            array_agg(train_mean_squared_error_avg) train_mean_squared_error_avg,
            array_agg(train_mean_squared_error_stddev) train_mean_squared_error_stddev,
            
            array_agg(test_kullback_leibler_divergence_avg) test_kullback_leibler_divergence_avg,
            array_agg(test_kullback_leibler_divergence_stddev) test_kullback_leibler_divergence_stddev,
            array_agg(train_kullback_leibler_divergence_avg) train_kullback_leibler_divergence_avg,
            array_agg(train_kullback_leibler_divergence_stddev) train_kullback_leibler_divergence_stddev
        from (
            select 
                (COUNT(test_loss_nan))::smallint num,
            
                (COUNT(test_loss))::smallint test_loss_num_finite,
                (AVG(test_loss))::real test_loss_avg,
                (stddev_samp(test_loss))::real test_loss_stddev,
                MIN(test_loss) test_loss_min,
                MAX(test_loss) test_loss_max,
                (PERCENTILE_DISC(array[.166, .333, .5, .667, .834]) WITHIN GROUP(ORDER BY test_loss_nan)) test_loss_percentile,
            
                (COUNT(train_loss))::smallint train_loss_num_finite,
                (AVG(train_loss))::real train_loss_avg,
                (stddev_samp(train_loss))::real train_loss_stddev,
                MIN(train_loss) train_loss_min,
                MAX(train_loss) train_loss_max,
                (PERCENTILE_DISC(array[.166, .333, .5, .667, .834]) WITHIN GROUP(ORDER BY train_loss_nan)) train_loss_percentile,
            
                (AVG(test_accuracy))::real test_accuracy_avg,
                (stddev_samp(test_accuracy))::real test_accuracy_stddev,
                (AVG(train_accuracy))::real train_accuracy_avg,
                (stddev_samp(train_accuracy))::real train_accuracy_stddev,
                (AVG(test_mean_squared_error))::real test_mean_squared_error_avg,
                (stddev_samp(test_mean_squared_error))::real test_mean_squared_error_stddev,
                (AVG(train_mean_squared_error))::real train_mean_squared_error_avg,
                (stddev_samp(train_mean_squared_error))::real train_mean_squared_error_stddev,
                
                (AVG(test_kullback_leibler_divergence))::real test_kullback_leibler_divergence_avg,
                (stddev_samp(test_kullback_leibler_divergence))::real test_kullback_leibler_divergence_stddev,
                (AVG(train_kullback_leibler_divergence))::real train_kullback_leibler_divergence_avg,
                (stddev_samp(train_kullback_leibler_divergence))::real train_kullback_leibler_divergence_stddev
            from (
                select
                    epoch::smallint epoch, 
                    v test_loss,
                    COALESCE(v, 'NaN'::real) test_loss_nan,
                    train_loss[epoch] train_loss,
                    COALESCE(train_loss[epoch], 'NaN'::real) train_loss_nan,
                
                    test_accuracy[epoch] test_accuracy,
                    train_accuracy[epoch] train_accuracy,
                    test_mean_squared_error[epoch] test_mean_squared_error,
                    train_mean_squared_error[epoch] train_mean_squared_error,
                    
                    test_kullback_leibler_divergence[epoch] test_kullback_leibler_divergence,
                    train_kullback_leibler_divergence[epoch] train_kullback_leibler_divergence
                from
                    run r,
                    unnest(r.test_loss) WITH ORDINALITY as epoch_value(v, epoch)
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



update job_status s
set status = 0, start_time = NULL, update_time = NULL
where queue = 1 and status = 1 and (now()::timestamp - start_time) > '2880 minutes';

select
    r.slurm_job_id,
    r.job_id,
    r.hostname,
    r.platform,
    s.start_time,
    s.update_time,
    (s.update_time - s.start_time) run_time,
    s.worker worker_id,
    command->'batch' batch,
    command->'size' size,
    command->'shape' shape,
    command->'depth' depth,
    (select string_value from parameter_ p where p.kind='tensorflow_version' and r.run_parameters @> array[p.id] limit 1) tensorflow_version
from
    run_ r,
    job_status s,
    job_data d
where
    r.job_id = s.id
    and s.id = d.id
    and s.queue = 1
    and s.status = 2
order by s.update_time desc
limit 100;

select
    s.start_time,
    s.update_time,
    (s.update_time - s.start_time) run_time,
    s.worker worker_id,
    command->'batch' batch,
    command->'size' size,
    command->'shape' shape,
    command->'depth' depth,
    s.error_count,
    s.error    
from
    job_status s,
    job_data d
where
    s.id = d.id
    and s.queue = 1
    and s.status = 3
order by s.update_time desc
limit 1000;



select
    inter,
    (3600 * num_runs / inter_seconds) run_rate,
    (3600 * effort / inter_seconds)::bigint effort_throughput,
    (3600 * num_runs / worker_seconds) worker_throughput,
    (3600 * effort / worker_seconds)::bigint worker_effort_throughput,
    (3600 * num_runs / node_seconds) node_run_throughput,
    (3600 * effort / node_seconds)::bigint node_effort_throughput,
    num_nodes,
    num_workers,
    num_runs,
    worker_runtime,
    node_runtime,
    effort,
    num_workers / num_nodes worker_to_node_ratio,
    worker_seconds / node_seconds worker_to_node_time,
    core_runtime,
    gpu_runtime
from
    (
    select
        inter,
        count(*) num_nodes,
        sum(num_workers) num_workers,
        sum(num_runs) num_runs,
        EXTRACT(epoch FROM sum(worker_runtime)) * interval '1 sec' worker_runtime,
        EXTRACT(epoch FROM sum(node_runtime)) * interval '1 sec' node_runtime,
        sum(effort) effort,
        EXTRACT(epoch FROM sum(node_runtime)) node_seconds,
        EXTRACT(epoch FROM sum(worker_runtime)) worker_seconds,
        EXTRACT(epoch FROM inter) inter_seconds,
        EXTRACT(epoch FROM sum(core_runtime)) * interval '1 sec' core_runtime,
        EXTRACT(epoch FROM sum(gpu_runtime)) * interval '1 sec' gpu_runtime
    from
        (
        select
            host.*,
         (CASE
                -- eagle gpu node
                WHEN hostname like 'r103u%' THEN node_runtime * 38
                -- eagle compute node
                WHEN (hostname like 'r%' and (not hostname like 'r103%')) THEN node_runtime * 38
                --vermillion 
                WHEN hostname like 'vs-gpu%' THEN node_runtime * 30
                WHEN hostname like 'vs-std%' THEN node_runtime * 30
                WHEN hostname like 'vs-lg%' THEN node_runtime * 60
                -- perlmutter gpu
                WHEN (hostname like 'nid001%' or hostname like 'nid002%' or hostname like 'nid003%' or hostname like 'nid0040%') THEN node_runtime * 128
                -- perlmutter cpu
                WHEN hostname like 'nid0048%' or hostname like 'nid005%' THEN node_runtime * 256
          ELSE node_runtime * 38
          END) core_runtime,
            (CASE
                -- eagle gpu node
                WHEN hostname like 'r103u%' THEN node_runtime * 2
                -- eagle compute node
                WHEN hostname like 'r%' and not hostname like 'r103%' THEN node_runtime * 0
                --vermillion 
                WHEN hostname like 'vs-gpu%' THEN node_runtime * 1
                WHEN hostname like 'vs-std%' THEN node_runtime * 0
                WHEN hostname like 'vs-lg%' THEN node_runtime * 0
                -- perlmutter gpu
                WHEN (hostname like 'nid001%' or hostname like 'nid002%' or hostname like 'nid003%' or hostname like 'nid0040%') THEN node_runtime * 4
                -- perlmutter cpu
                WHEN hostname like 'nid0048%' or hostname like 'nid005%' THEN node_runtime * 0
          ELSE node_runtime * 0
          END) gpu_runtime
        FROM
            (
        select
            inter,
            hostname,
            count(*) num_workers,
            sum(num_runs) num_runs,
            sum(worker_runtime) worker_runtime,
            sum(effort) effort,
            max(last_finish) - min(first_start) node_runtime
        from
            (
            select 
                inter,
                slurm_job_id,
                hostname,
                worker,
                count(*) num_runs,
                sum(run_time) worker_runtime,
                sum(effort) effort,
                max(update_time) last_finish,
                min(start_time) first_start
            from
                (
                     select
                        inter,
                        r.slurm_job_id,
                        r.hostname,
                        r.platform,
                        s.start_time,
                        s.update_time,
                        (s.update_time - s.start_time) run_time,
                        s.worker worker,
                        command->'batch' batch,
                        (command->'size')::bigint * (command->'run_config'->'epochs')::bigint effort,
                        command->'shape' shape,
                        command->'depth' depth
                    from
                        (
                            select 
                                inter::interval inter,
                                (now()::timestamp -  inter::interval) start_limit
                            from (VALUES ('3 hours'), ('6 hours'), ('12 hours'), ('1 day'), ('2 days'), ('4 days'), ('8 days'), ('16 days'),
                                ('30 days'), ('60 days'), ('120 days'), ('1 year'), ('2 year')
                                ) inter (inter)
                        ) inter,
                        run_ r,
                        job_status s,
                        job_data d
                    where
                        s.start_time >= inter.start_limit
                        and r.job_id = s.id
                        and s.id = d.id
                        and s.status = 2
--                         and s.queue = 1
                        and (s.update_time - s.start_time) >= '1 second'
                 ) j
            group by inter, slurm_job_id, hostname, worker
            ) worker
        group by inter, slurm_job_id, hostname
        ) host
        ) host
    group by inter
    ) agg
order by inter asc
;






update experiment_ e set 
    "size" = size_.integer_value,
    "relative_size_error" = (abs( (size_.integer_value - e.num_free_parameters) / (size_.integer_value)::float))::real
FROM
parameter_ as size_
WHERE
(e.size is NULL OR e.relative_size_error is NULL) and
e.experiment_parameters @> array[size_.id] and size_.kind = 'size';



select * from parameter_ order by kind, string_value, integer_value, real_value, bool_value;

-- check queue status

select 
    queue, command->'batch' batch, status, count(*) num, min(priority) min_priority, max(priority) max_priority
from 
    job_status s,
    job_data d
where s.id = d.id and queue = 2 and status IN (0,1,2,3)
group by queue, batch, status
order by queue, batch, status asc, min_priority asc;


select 
    queue, status, min(priority) min_priority, max(priority) max_priority, count(*) num, command->'batch' batch, command->'shape' shape, command->'dataset' dataset
from 
    job_status s,
    job_data d
where s.id = d.id and queue = 1 and status IN (0,1,2,3)
group by status, queue, batch, shape, dataset
order by status asc, min_priority asc, queue, batch, shape, dataset;

(select 
    queue, status, min(priority) min_priority, max(priority) max_priority, count(*) num, command->'batch' batch, command->'shape' shape, command->'dataset' dataset, max(update_time) last_update
from 
    job_status s,
    job_data d
where s.id = d.id and queue = 1 and status = 0 --and status IN (0,1,3)
group by status, queue, batch, shape, dataset
order by status asc, min_priority asc, queue, batch, shape, dataset)
union all
(select 
    queue, status, min(priority) min_priority, max(priority) max_priority, count(*) num, command->'batch' batch, command->'shape' shape, command->'dataset' dataset, max(update_time) last_update
from 
    job_status s,
    job_data d
where s.id = d.id and queue = 1 and status > 0 and status < 4
group by status, queue, batch, shape, dataset
order by status asc, queue, batch, shape, dataset
);

(select 
    queue, 0 status_, min(priority) min_priority, max(priority) max_priority, count(*) num, command->'batch' batch, command->'shape' shape, command->'dataset' dataset, 
--     command->'depth' depth, 
--     command->'optimizer'->'class_name' optimizer,
--     command->'optimizer'->'config'->'learning_rate' learning_rate,
--     command->'run_config'->'batch_size' batch_size,
    max(update_time) last_update
from 
    job_status s,
    job_data d
where s.id = d.id and queue = 1 and status = 0 --and status IN (0,1,3)
group by status_, queue, batch, shape, dataset
order by status_ asc, min_priority asc, queue, batch, shape, dataset)
union all
(select 
    queue, (CASE 
            WHEN status = 1 THEN 1
            ELSE 2
            END
           ) status_, min(priority) min_priority, max(priority) max_priority, count(*) num, command->'batch' batch, command->'shape' shape, command->'dataset' dataset, 
--             command->'depth' depth, 
--             command->'optimizer'->'class_name' optimizer,
--             command->'optimizer'->'config'->'learning_rate' learning_rate,
--             command->'run_config'->'batch_size' batch_size,
            max(update_time) last_update
from 
    job_status s,
    job_data d
where s.id = d.id and queue = 1 and status > 0
group by status_, queue, batch, shape, dataset
order by status_ asc, queue, batch, shape, dataset
);


select command->'batch' batch, command->'dataset' dataset, command->'size' size, command->'depth' depth, command->'shape' shape, command->'optimizer'->'class_name' optimizer,
            command->'optimizer'->'config'->'learning_rate' learning_rate, command->'run_config'->'batch_size' batch_size,  s.*, d.* 
from job_status s inner join job_data d on (s.id = d.id)
where queue = 1 and status = 0 order by priority asc limit 1000;

select command->'batch' batch, command->'dataset' dataset, command->'size' size, command->'depth' depth, command->'shape' shape, command->'optimizer'->'class_name' optimizer,
            command->'optimizer'->'config'->'learning_rate' learning_rate, command->'run_config'->'batch_size' batch_size,  s.*, d.* 
from job_status s inner join job_data d on (s.id = d.id)
where queue = 1 and status = 1 order by update_time desc limit 1000;

select  (now()-update_time) age, (update_time - start_time) runtime, r.hostname, r.platform, r.slurm_job_id, tensorflow_version.string_value tensorflow_version, python_version.string_value python_version, (command->'size')::bigint * (command->'run_config'->'epochs')::bigint effort, ((command->'size')::bigint * (command->'run_config'->'epochs')::bigint / EXTRACT(epoch FROM (update_time-start_time)))::bigint effort_rate, command->'batch' batch, command->'dataset' dataset, command->'size' size, command->'depth' depth, command->'shape' shape, command->'optimizer'->'class_name' optimizer,
            command->'optimizer'->'config'->'learning_rate' learning_rate, command->'run_config'->'batch_size' batch_size,  s.*, d.* 
from job_status s inner join job_data d on (s.id = d.id) left outer join run_ r on (s.id = r.run_id) 
left outer join parameter_ tensorflow_version on (tensorflow_version.kind='tensorflow_version' and r.run_parameters @> array[tensorflow_version.id])
left outer join parameter_ python_version on (python_version.kind='python_version' and r.run_parameters @> array[python_version.id])
where queue = 1 and status = 2 order by update_time desc limit 1000;


select(now()-update_time) age, (update_time - start_time) runtime,  command->'batch' batch, command->'dataset', command->'size', command->'shape' shape, (update_time - start_time) runtime, * 
from job_status s inner join job_data d on (s.id = d.id) 
where queue = 1 and status = 3 order by update_time desc limit 1000;

update job_status s
set status = 0, start_time = NULL, update_time = NULL
where queue = 1 and status = 1 and (now()::timestamp - start_time) > '1 days';

select * from job_status s
where queue = 1 and status = 3
order by update_time desc
limit 4000;

select error, count(error) num from job_status s
where queue = 1 and status = 3
-- order by update_time desc
group by error
order by num desc
limit 4000;

update job_status s
set status = 0, start_time = NULL, update_time = NULL
where 
    queue = 1 and status = 3
    and s.error not like 'Could not find%'
;

update job_status s
set status = 4
where 
    queue = 1 and status = 3
    and s.error like 'Could not find%'
;

select
    s.start_time,
    s.update_time,
    (s.update_time - s.start_time) run_time,
    s.worker worker_id,
    command->'batch' batch,
    command->'size' size,
    command->'shape' shape,
    command->'depth' depth,
    command->'dataset' dataset,
    s.error_count,
    s.error    
from
    job_status s,
    job_data d
where
    s.id = d.id
    and s.queue = 1
    and s.status = 3
    and s.error not like 'Could not find%'
    and s.error not like ' BiasGrad %'
    and s.error not like '%RESOURCE_EXHAUSTED%'
    and s.error not like ' OOM%'
order by s.update_time desc
limit 1000;

-- reset group priority
update job_status stat
    set priority = priority + (7300000 -  22300001)
from
(
    select ROW_NUMBER() OVER() seq, ordered_job.id id, sctid
    from
    (   select 
        (command->'depth')::bigint depth,
        (command->'kernel_regularizer'->'l2')::float lambda,
        (command->'seed')::bigint seed,
        (command->>'dataset') dataset,
        s.id,
        s.ctid sctid
        from
            job_status s,
            job_data d
        where
            s.id = d.id and queue = 1 and status IN (0,1,3)
            and command @> jsonb_build_object(
                'batch', 'l2_group_1')
--                 'shape', 'trapezoid')
--             and command->>'shape' in (
--                 'rectangle',
--                 'exponential',
--                 'trapezoid',
--                 'wide_first_2x',
--                 'rectangle_residual'
--             )
--         order by depth asc, lambda asc, dataset, floor(random() * 100)::smallint asc
        order by priority asc
        for update
    ) ordered_job
) seq
where
seq.sctid = stat.ctid
;

-- compact priority queue:
update job_status stat
    set priority = seq.seq
from
(
select 
    job.sctid,
    job.id,
    group_num * 1000 + element_num seq
from
    (
        select
            ROW_NUMBER() OVER() group_num,
            grp.*
        from
            (
            select 
                min(priority) min_priority, command->'batch' batch, 
                command->'shape' shape, 
                command->'dataset' dataset,
                command->'depth' depth, 
                command->'optimizer'->'class_name' optimizer,
                command->'optimizer'->'config'->'learning_rate' learning_rate,
                command->'run_config'->'batch_size' batch_size
            from 
                job_status s,
                job_data d
            where s.id = d.id and queue = 1 and status IN (0,1,3)
            group by batch, shape, depth, dataset, optimizer, learning_rate, batch_size
            order by min_priority asc
            ) grp
    ) pq,
    lateral (
        select 
            ROW_NUMBER() OVER() element_num,
            job.id,
            job.sctid
        from
            (
            select s.id, priority, s.ctid sctid
            from
                job_status s,
                job_data d
            where
                s.id = d.id and queue = 1 and status IN (0,1,3)
                and command @> jsonb_build_object(
                    'batch', pq.batch,
                    'shape', pq.shape,
                    'dataset', pq.dataset,
                    'depth', pq.depth,
                    'optimizer', jsonb_build_object('class_name', pq.optimizer, 'config', jsonb_build_object( 'learning_rate', pq.learning_rate)),
                    'run_config', jsonb_build_object('batch_size', pq.batch_size)
                )::jsonb
            order by priority asc
            ) job
    ) job
) seq
where
stat.ctid = seq.sctid
;

update job_status stat
    set priority = seq + 2000000
from
(
    select ROW_NUMBER() OVER() seq, ordered_job.id id, sctid
    from
    (   
        select * from ( select floor( group_sequence / 5) group_number, * from (select (row_number() over( partition by depth, dataset, optimizer, learning_rate, batch_size)) group_sequence, * from (
        select 
            (command->'depth')::bigint depth,
            (command->'seed')::bigint seed,
            (command->>'dataset') dataset,
            (command->'optimizer'->'class_name')::text optimizer,
            (command->'optimizer'->'config'->'learning_rate')::real learning_rate,
            (command->'run_config'->'batch_size')::bigint batch_size,
            s.id,
            s.ctid sctid
        from
            job_status s,
            job_data d
        where
            s.id = d.id and queue = 1 and status IN (0,1,2,3)
--             and command->>'shape' like '%x_residual'
            and command @> jsonb_build_object(
                'batch', 'optimizer_1')
--             and command->>'shape' like '%x_residual'
--                 'shape', 'trapezoid')
--             and command->>'shape' in (
--                 'rectangle',
--                 'exponential',
--                 'trapezoid',
--                 'wide_first_2x',
--                 'rectangle_residual'
--             )
         for update
        ) x ) x ) x
        order by group_number asc, abs(depth-3.1) asc, abs(log(batch_size) - log(256 - .1)) asc, abs(log(learning_rate) - log(0.00011)) asc, group_sequence asc, floor(random() * 16000)::smallint 
    ) ordered_job
) seq
where
seq.sctid = stat.ctid
;


-- May 3rd, 2022, 6:46pm -> '2022-05-03T18:46:00'
-- ((date_part('epoch'::text, CURRENT_TIMESTAMP) - (1600000000)::double precision))::integer

--- change params
WITH from_param as
(
    select * from parameter_ where (kind='kernel_regularizer.l1' and real_value = 0) or (kind='kernel_regularizer.l2' and real_value = 0)
),
to_param as (
    select "id" from parameter_ where kind = 'kernel_regularizer' and real_value is null limit 1
),
exp_edit as (
    select 
        widths,
        e.experiment_id experiment_id,
        new_params.new_params new_params,
        e.experiment_parameters old_params,
        (select x.experiment_id from experiment_ x where x.experiment_parameters = new_params.new_params for update) merge_into
    from 
        (
            select * from experiment_ e, from_param
            where e.experiment_parameters @> array[(select id from from_param)]
            for update
        ) e,
        lateral (
            select array_agg(id)::smallint[] new_params from
            (
                select distinct id from
                    (
                    select id from
                        unnest(e.experiment_parameters) as id
                    where
                        id not in (select id from from_param)
                    union all
                    select id from to_param
                    ) x
                order by id asc
            ) x
        ) new_params
),
exp_to_update as (
    select * from exp_edit where merge_into is NULL
),
exp_to_merge as
(
    select * from exp_edit where merge_into is not NULL
),
updated_runs as (
    update run_ r set
        experiment_id = COALESCE(exp_edit.merge_into, exp_edit.experiment_id),
        run_parameters = (
            select array_agg(id)::smallint[] run_parameters from
                (
                    select distinct id from
                        (
                        select id from
                            unnest(r.run_parameters) as id
                        where
                            id not in (select id from from_param)
                        union all
                        select id from to_param
                        ) x
                    order by id asc
                ) x
        )
    from
        exp_edit
    where
        exp_edit.experiment_id = r.experiment_id
),
updated_exp as (
    update experiment_ e
        set experiment_parameters = exp_to_update.new_params
    from
        exp_to_update
    where
        exp_to_update.experiment_id = e.experiment_id
),
updated_summary as (
    update experiment_summary_ e
        set experiment_parameters = exp_to_update.new_params
    from
        exp_to_update
    where
        exp_to_update.experiment_id = e.experiment_id
),
deleted_exp as (
    delete from experiment_ e
    using
        exp_to_merge
    where
        exp_to_merge.experiment_id = e.experiment_id
),
deleted_summary as (
    delete from experiment_summary_ e
    using
        exp_to_merge
    where
        exp_to_merge.experiment_id = e.experiment_id
)
select (select count(*) from exp_edit), (select count(*) from exp_to_merge), (select count(*) from exp_to_update)
;


WITH shape_params as
(
    select * from parameter_ shape where (shape.kind = 'shape' and shape.string_value IN ('wide_first_4x', 'wide_first_8x', 'wide_first_16x'))
),
wide_first_2x_parameter as (
    select "id" from parameter_ where kind = 'shape' and string_value = 'wide_first_2x' limit 1
),
exp_to_update0 as (
    select 
        *
--         (CASE WHEN merge_into is NULL THEN (nextval('experiment__experiment_id_seq1'::regclass)) ELSE NULL END) new_experiment_id
--         (CASE WHEN merge_into is NULL THEN -1 ELSE NULL END) new_experiment_id
    FROM
    (
        select 
            widths,
    --         shape.string_value shape,    
    --         depth.integer_value depth,
    --         size.integer_value size,
    --         dataset.string_value dataset,
    --         batch.string_value batch,
            e.experiment_id experiment_id,
            new_params.experiment_parameters new_params,
            e.experiment_parameters old_params
--             (select tst.experiment_id from experiment_ tst where tst.experiment_parameters = new_params.experiment_parameters limit 1 for update) merge_into
        from 
            experiment_ e,
            shape_params shape,
    --         parameter_ shape,
    --         parameter_ depth,
    --         parameter_ size,
    --         parameter_ dataset,
    --         parameter_ batch,
            lateral (
                select array_agg(id)::smallint[] experiment_parameters from
                (
                    select distinct id from
                        (
                        select id from
                            unnest(e.experiment_parameters) as id
                        where
                            id not in (select id from shape_params)
                        union all
                        select id from wide_first_2x_parameter
                        ) x
                    order by id asc
                ) x
            ) new_params
        where 
           e.experiment_parameters @> array[shape.id]
    --                 shape.id in (select id from shape_params) and e.experiment_parameters @> array[shape.id]
                    -- and (shape.kind = 'shape' and shape.string_value IN ('wide_first_4x', 'wide_first_8x', 'wide_first_16x') and e.experiment_parameters @> array[shape.id])
    --                 and (depth.kind = 'depth' and e.experiment_parameters @> array[depth.id])
    --                 and (size.kind = 'size' and e.experiment_parameters @> array[size.id])
    --                 and (dataset.kind = 'dataset' and e.experiment_parameters @> array[dataset.id])
    --                 and (batch.kind = 'batch' and e.experiment_parameters @> array[batch.id])
                    -- and (widths[1]::float / widths[2]) <= 3.5
        ) x
),
exp_to_insert as (
    select * from exp_to_update0 where not exists (select * from experiment_ e where e.experiment_parameters = exp_to_update0.new_params)
    for update
),
ir as (
    insert into experiment_ (
        experiment_parameters
        )
    select
--         new_experiment_id experiment_id, --experiment_id integer NOT NULL DEFAULT nextval('experiment__experiment_id_seq1'::regclass),
        new_params as experiment_parameters -- experiment_parameters smallint[] NOT NULL,
--         NULL num_free_parameters,
--         NULL network_structure,
--         NULL widths,
--         NULL size,
--         NULL relative_size_errror
    FROM
        exp_to_insert
    ON CONFLICT DO NOTHING
    RETURNING 
        experiment_id, 
        experiment_parameters
),
exp_to_update as (
    select 
        exp_to_update0.*, 
        COALESCE (
            (select experiment_id from ir where ir.experiment_parameters = exp_to_update0.new_params limit 1),
            (select experiment_id from experiment_ e where e.experiment_parameters = exp_to_update0.new_params limit 1)
        ) merge_into 
    from exp_to_update0, experiment_ e 
    where e.experiment_parameters = exp_to_update0.new_params
        and not exists (select * from ir where ir.experiment_parameters = exp_to_update0.new_params)
),
runs_to_update as (
    update run_ r set
        experiment_id = merge_into,
        run_parameters = (
            select array_agg(id)::smallint[] run_parameters from
                (
                    select distinct id from
                        (
                        select id from
                            unnest(r.run_parameters) as id
                        where
                            id not in (select id from shape_params)
                        union all
                        select id from wide_first_2x_parameter
                        ) x
                    order by id asc
                ) x
        )
--     select
--         (COALESCE (exp_to_update.merge_into, exp_to_update.new_experiment_id)) as experiment_id,
--         (
--             select array_agg(id)::smallint[] run_parameters from
--                 (
--                     select distinct id from
--                         (
--                         select id from
--                             unnest(r.run_parameters) as id
--                         where
--                             id not in (select id from shape_params)
--                         union all
--                         select id from wide_first_2x_parameter
--                         ) x
--                     order by id asc
--                 ) x
--         ) run_parameters
    from
        exp_to_update,
--         run_ r,
        job_status s
    where
        exp_to_update.experiment_id = r.experiment_id
        and s.id = r.job_id
        and r.record_timestamp <= ((date_part('epoch'::text, '2022-05-04T18:46:00'::timestamp) - (1600000000)::double precision))::integer
        and s.update_time <= '2022-05-03T18:47:00'::timestamp
),
summaries_to_delete as (
--     select
--         s.experiment_id
    delete from experiment_summary_ s
    using exp_to_update
--     from
--         exp_to_update,
--         experiment_summary_ s
    where
        s.experiment_id = exp_to_update.experiment_id
)
select (select count(*) from exp_to_update)
;


--- summary update with min points:
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
    train_loss_num_finite, train_loss_min, train_loss_max,
    train_loss_avg, train_loss_stddev,
    train_loss_q1, train_loss_median, train_loss_q3,
    test_accuracy_avg, test_accuracy_stddev,
    test_accuracy_q1, test_accuracy_median, test_accuracy_q3,
    train_accuracy_avg, train_accuracy_stddev,
    train_accuracy_q1, train_accuracy_median, train_accuracy_q3,
    test_mean_squared_error_avg, test_mean_squared_error_stddev,
    test_mean_squared_error_q1, test_mean_squared_error_median, test_mean_squared_error_q3,
    train_mean_squared_error_avg, train_mean_squared_error_stddev,
    train_mean_squared_error_q1, train_mean_squared_error_median, train_mean_squared_error_q3,
    test_kullback_leibler_divergence_avg, test_kullback_leibler_divergence_stddev,
    test_kullback_leibler_divergence_q1, test_kullback_leibler_divergence_median, test_kullback_leibler_divergence_q3,
    train_kullback_leibler_divergence_avg, train_kullback_leibler_divergence_stddev,
    train_kullback_leibler_divergence_q1, train_kullback_leibler_divergence_median, train_kullback_leibler_divergence_q3,
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
    train_loss_num_finite, train_loss_min, train_loss_max,
    train_loss_avg, train_loss_stddev,
    train_loss_q1, train_loss_median, train_loss_q3,
    test_accuracy_avg, test_accuracy_stddev,
    test_accuracy_q1, test_accuracy_median, test_accuracy_q3,
    train_accuracy_avg, train_accuracy_stddev,
    train_accuracy_q1, train_accuracy_median, train_accuracy_q3,
    test_mean_squared_error_avg, test_mean_squared_error_stddev,
    test_mean_squared_error_q1, test_mean_squared_error_median, test_mean_squared_error_q3,
    train_mean_squared_error_avg, train_mean_squared_error_stddev,
    train_mean_squared_error_q1, train_mean_squared_error_median, train_mean_squared_error_q3,
    test_kullback_leibler_divergence_avg, test_kullback_leibler_divergence_stddev,
    test_kullback_leibler_divergence_q1, test_kullback_leibler_divergence_median, test_kullback_leibler_divergence_q3,
    train_kullback_leibler_divergence_avg, train_kullback_leibler_divergence_stddev,
    train_kullback_leibler_divergence_q1, train_kullback_leibler_divergence_median, train_kullback_leibler_divergence_q3,
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
    order by experiment_id offset 0 limit 1000
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
            array_agg(v_num_finite) train_loss_num_finite, array_agg(v_min) train_loss_min, array_agg(v_max) train_loss_max,
            array_agg(v_avg) train_loss_avg, array_agg(v_stddev) train_loss_stddev,
            array_agg(v_percentile[1]) train_loss_q1, array_agg(v_percentile[2]) train_loss_median, array_agg(v_percentile[3]) train_loss_q3
        from
        (       
            select
                (COUNT(v))::smallint v_num_finite, MIN(v)::real v_min, MAX(v)::real v_max,    
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev,
                (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
            from
                rr inner join run_ r on rr.run_ctid = r.ctid,
                unnest(r.train_loss) WITH ORDINALITY as epoch_value(v, epoch)
            group by epoch order by epoch
        ) x
    ) train_loss_,
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
    lateral (select array_agg(v_avg) train_accuracy_avg, array_agg(v_stddev) train_accuracy_stddev, array_agg(v_percentile[1]) train_accuracy_q1, array_agg(v_percentile[2]) train_accuracy_median, array_agg(v_percentile[3]) train_accuracy_q3 from
            (select
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.train_accuracy) WITH ORDINALITY as epoch_value(v, epoch)
                group by epoch order by epoch
            ) x ) train_accuracy_,
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
    lateral (select array_agg(v_avg) train_mean_squared_error_avg, array_agg(v_stddev) train_mean_squared_error_stddev, array_agg(v_percentile[1]) train_mean_squared_error_q1, array_agg(v_percentile[2]) train_mean_squared_error_median, array_agg(v_percentile[3]) train_mean_squared_error_q3 from
            (select
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.train_mean_squared_error) WITH ORDINALITY as epoch_value(v, epoch)
                group by epoch order by epoch
            ) x ) train_mean_squared_error_,
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
    lateral (select array_agg(v_avg) train_kullback_leibler_divergence_avg, array_agg(v_stddev) train_kullback_leibler_divergence_stddev, array_agg(v_percentile[1]) train_kullback_leibler_divergence_q1, array_agg(v_percentile[2]) train_kullback_leibler_divergence_median, array_agg(v_percentile[3]) train_kullback_leibler_divergence_q3 from
            (select
                (AVG(v))::real v_avg, (stddev_samp(v))::real v_stddev, (PERCENTILE_CONT(array[.25, .5, .75]) WITHIN GROUP(ORDER BY coalesce(v, 'NaN'::real)))::real[] v_percentile
                from rr inner join run_ r on rr.run_ctid = r.ctid, unnest(r.train_kullback_leibler_divergence) WITH ORDINALITY as epoch_value(v, epoch)
                group by epoch order by epoch
            ) x ) train_kullback_leibler_divergence_
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
        train_loss_num_finite = EXCLUDED.train_loss_num_finite,train_loss_min = EXCLUDED.train_loss_min,train_loss_max = EXCLUDED.train_loss_max,
        train_loss_avg = EXCLUDED.train_loss_avg,train_loss_stddev = EXCLUDED.train_loss_stddev,
        train_loss_q1 = EXCLUDED.train_loss_q1,train_loss_median = EXCLUDED.train_loss_median,train_loss_q3 = EXCLUDED.train_loss_q3,
        test_accuracy_avg = EXCLUDED.test_accuracy_avg,test_accuracy_stddev = EXCLUDED.test_accuracy_stddev,
        test_accuracy_q1 = EXCLUDED.test_accuracy_q1,test_accuracy_median = EXCLUDED.test_accuracy_median,test_accuracy_q3 = EXCLUDED.test_accuracy_q3,
        train_accuracy_avg = EXCLUDED.train_accuracy_avg,train_accuracy_stddev = EXCLUDED.train_accuracy_stddev,
        train_accuracy_q1 = EXCLUDED.train_accuracy_q1,train_accuracy_median = EXCLUDED.train_accuracy_median,train_accuracy_q3 = EXCLUDED.train_accuracy_q3,
        test_mean_squared_error_avg = EXCLUDED.test_mean_squared_error_avg,test_mean_squared_error_stddev = EXCLUDED.test_mean_squared_error_stddev,
        test_mean_squared_error_q1 = EXCLUDED.test_mean_squared_error_q1,test_mean_squared_error_median = EXCLUDED.test_mean_squared_error_median,test_mean_squared_error_q3 = EXCLUDED.test_mean_squared_error_q3,
        train_mean_squared_error_avg = EXCLUDED.train_mean_squared_error_avg,train_mean_squared_error_stddev = EXCLUDED.train_mean_squared_error_stddev,
        train_mean_squared_error_q1 = EXCLUDED.train_mean_squared_error_q1,train_mean_squared_error_median = EXCLUDED.train_mean_squared_error_median,train_mean_squared_error_q3 = EXCLUDED.train_mean_squared_error_q3,
        test_kullback_leibler_divergence_avg = EXCLUDED.test_kullback_leibler_divergence_avg,test_kullback_leibler_divergence_stddev = EXCLUDED.test_kullback_leibler_divergence_stddev,
        test_kullback_leibler_divergence_q1 = EXCLUDED.test_kullback_leibler_divergence_q1,test_kullback_leibler_divergence_median = EXCLUDED.test_kullback_leibler_divergence_median,test_kullback_leibler_divergence_q3 = EXCLUDED.test_kullback_leibler_divergence_q3,
        train_kullback_leibler_divergence_avg = EXCLUDED.train_kullback_leibler_divergence_avg,train_kullback_leibler_divergence_stddev = EXCLUDED.train_kullback_leibler_divergence_stddev,
        train_kullback_leibler_divergence_q1 = EXCLUDED.train_kullback_leibler_divergence_q1,train_kullback_leibler_divergence_median = EXCLUDED.train_kullback_leibler_divergence_median,train_kullback_leibler_divergence_q3 = EXCLUDED.train_kullback_leibler_divergence_q3,
        test_loss_min_epoch_min = EXCLUDED.test_loss_min_epoch_min, test_loss_min_epoch_max = EXCLUDED.test_loss_min_epoch_max, test_loss_min_epoch_avg = EXCLUDED.test_loss_min_epoch_avg, test_loss_min_epoch_q1 = EXCLUDED.test_loss_min_epoch_q1, test_loss_min_epoch_median = EXCLUDED.test_loss_min_epoch_median, test_loss_min_epoch_q3 = EXCLUDED.test_loss_min_epoch_q3, test_loss_min_value_min = EXCLUDED.test_loss_min_value_min, test_loss_min_value_max = EXCLUDED.test_loss_min_value_max, test_loss_min_value_avg = EXCLUDED.test_loss_min_value_avg, test_loss_min_value_q1 = EXCLUDED.test_loss_min_value_q1, test_loss_min_value_median = EXCLUDED.test_loss_min_value_median, test_loss_min_value_q3 = EXCLUDED.test_loss_min_value_q3, test_accuracy_max_epoch_min = EXCLUDED.test_accuracy_max_epoch_min, test_accuracy_max_epoch_max = EXCLUDED.test_accuracy_max_epoch_max, test_accuracy_max_epoch_avg = EXCLUDED.test_accuracy_max_epoch_avg, test_accuracy_max_epoch_q1 = EXCLUDED.test_accuracy_max_epoch_q1, test_accuracy_max_epoch_median = EXCLUDED.test_accuracy_max_epoch_median, test_accuracy_max_epoch_q3 = EXCLUDED.test_accuracy_max_epoch_q3, test_accuracy_max_value_min = EXCLUDED.test_accuracy_max_value_min, test_accuracy_max_value_max = EXCLUDED.test_accuracy_max_value_max, test_accuracy_max_value_avg = EXCLUDED.test_accuracy_max_value_avg, test_accuracy_max_value_q1 = EXCLUDED.test_accuracy_max_value_q1, test_accuracy_max_value_median = EXCLUDED.test_accuracy_max_value_median, test_accuracy_max_value_q3 = EXCLUDED.test_accuracy_max_value_q3, test_mean_squared_error_min_epoch_min = EXCLUDED.test_mean_squared_error_min_epoch_min, test_mean_squared_error_min_epoch_max = EXCLUDED.test_mean_squared_error_min_epoch_max, test_mean_squared_error_min_epoch_avg = EXCLUDED.test_mean_squared_error_min_epoch_avg, test_mean_squared_error_min_epoch_q1 = EXCLUDED.test_mean_squared_error_min_epoch_q1, test_mean_squared_error_min_epoch_median = EXCLUDED.test_mean_squared_error_min_epoch_median, test_mean_squared_error_min_epoch_q3 = EXCLUDED.test_mean_squared_error_min_epoch_q3, test_mean_squared_error_min_value_min = EXCLUDED.test_mean_squared_error_min_value_min, test_mean_squared_error_min_value_max = EXCLUDED.test_mean_squared_error_min_value_max, test_mean_squared_error_min_value_avg = EXCLUDED.test_mean_squared_error_min_value_avg, test_mean_squared_error_min_value_q1 = EXCLUDED.test_mean_squared_error_min_value_q1, test_mean_squared_error_min_value_median = EXCLUDED.test_mean_squared_error_min_value_median, test_mean_squared_error_min_value_q3 = EXCLUDED.test_mean_squared_error_min_value_q3, test_kullback_leibler_divergence_min_epoch_min = EXCLUDED.test_kullback_leibler_divergence_min_epoch_min, test_kullback_leibler_divergence_min_epoch_max = EXCLUDED.test_kullback_leibler_divergence_min_epoch_max, test_kullback_leibler_divergence_min_epoch_avg = EXCLUDED.test_kullback_leibler_divergence_min_epoch_avg, test_kullback_leibler_divergence_min_epoch_q1 = EXCLUDED.test_kullback_leibler_divergence_min_epoch_q1, test_kullback_leibler_divergence_min_epoch_median = EXCLUDED.test_kullback_leibler_divergence_min_epoch_median, test_kullback_leibler_divergence_min_epoch_q3 = EXCLUDED.test_kullback_leibler_divergence_min_epoch_q3, test_kullback_leibler_divergence_min_value_min = EXCLUDED.test_kullback_leibler_divergence_min_value_min, test_kullback_leibler_divergence_min_value_max = EXCLUDED.test_kullback_leibler_divergence_min_value_max, test_kullback_leibler_divergence_min_value_avg = EXCLUDED.test_kullback_leibler_divergence_min_value_avg, test_kullback_leibler_divergence_min_value_q1 = EXCLUDED.test_kullback_leibler_divergence_min_value_q1, test_kullback_leibler_divergence_min_value_median = EXCLUDED.test_kullback_leibler_divergence_min_value_median, test_kullback_leibler_divergence_min_value_q3 = EXCLUDED.test_kullback_leibler_divergence_min_value_q3
;
