

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
       
       
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity 
WHERE usename = 'dmpappsops'
    AND query_start < (CURRENT_TIMESTAMP - '10s'::interval)
ORDER BY state, query_start desc;

update experiment_migration m set 
    is_valid = True,
    migrated = False,
    error_message = NULL
    WHERE EXISTS (select 1 from run_ r where r.experiment_id = m.experiment_id)
    AND NOT EXISTS (select 1 from experiment2 e where e.experiment_id = m.experiment_id)
    and error_message NOT LIKE 'failed on Could not find%' 
    and error_message NOT LIKE 'wrong number%wide_first_%'
    AND error_message NOT LIKE 'wrong number%poker%';
    
select * from attr where kind = 'model_shape';
alter table attr alter column value_json set storage EXTENDED;
ALTER TABLE attr SET (toast_tuple_target = 256)
select count(1) from experiment_migration where not migrated and is_valid;


    select
        *,
        (total-migrated) remaining,
        migrated::real / total pct_migrated,
        errored::real / migrated pct_errored
from
(
    select 
        count(1) total, 
        sum(migrated::integer) migrated,
        sum((migrated and error_message is not NULL)::integer) errored
    from experiment_migration where is_valid
    ) x;

select x.* from 
    experiment2 e,
    lateral (
        select 
            (   select value_str from unnest(experiment_attrs) ea(attr_id) inner join attr using(attr_id)
                where kind = 'dataset_name' limit 1
            ) dataset_name,
            (   select value_json from unnest(experiment_attrs) ea(attr_id) inner join attr using(attr_id)
                where kind = 'model_input_shape' limit 1
            ) model_input_shape,
            (   select value_int from unnest(experiment_attrs) ea(attr_id) inner join attr using(attr_id)
                where kind = 'model_output_units' limit 1
            ) model_output_shape
    ) x
where
    e.experiment_attrs && (
        select array_agg(attr_id) from attr
            where kind = 'dataset_name'
    )
group by dataset_name, model_input_shape, model_output_shape;


select 
    dataset_name, model_input_shape, model_output_units,
    count(distinct experiment_uid) num_exp,
    count(1) num_run
FROM (
select 
    e.experiment_uid,
    a_dataset_name.value_str dataset_name,
    a_model_input_shape.value_json model_input_shape,
    a_model_output_units.value_int model_output_units
from
    experiment2 e,
    run2 r,
    attr a_dataset_name,
    attr a_model_input_shape,
    attr a_model_output_units
where
    e.experiment_attrs && (
        select array_agg(attr_id) from attr
            where kind = 'dataset_name'
    )
    and r.experiment_uid = e.experiment_uid
    and a_dataset_name.kind = 'dataset_name'
    and e.experiment_attrs @> ARRAY[a_dataset_name.attr_id]
    and a_model_input_shape.kind = 'model_input_shape'
    and ARRAY[a_model_input_shape.attr_id] <@ e.experiment_attrs 
    and a_model_output_units.kind = 'model_output_units'
    and ARRAY[a_model_output_units.attr_id] <@ e.experiment_attrs
) x
group by dataset_name, model_input_shape, model_output_units;

select
    *
from
(
    select 
        sum(n) / count(1) compression_ratio,
        kind,
        value_type,
        mode() within group (order by v) mode_value,
        count(1) num_vals,
        avg(n) avg_num,
        min(n),
        percentile_cont(array[.1,.25,.5,.75,.9]) WITHIN GROUP (ORDER BY n),
        max(n)
    from
    (
        select 
            kind, 
            value_type,
            coalesce(
                to_jsonb(value_bool), 
                to_jsonb(value_int), 
                to_jsonb(value_float), 
                to_jsonb(value_str),
                to_jsonb(digest)
            ) v,
            count(1) n
        from attr left join experiment2 on (experiment_attributes @> array[attribute_id])
        group by kind, value_type, v
    ) ac
    group by kind, value_type
    order by kind, value_type
) x
order by compression_ratio asc;

select * from attr where kind like '%regul%';

select count(1) from experiment2 where experiment_attrs @> array[37];


select 
    e.*,
    x.*,
    json_object_agg(a.kind, coalesce(a.value_bool, a.value_int, a.value_float, a.value_str, a.value_json)) xattrib,
    json_object_agg(p.kind, coalesce(p.bool_value, p.int_value, p.real_value, p.string_value)) eattrib
from
    experiment_ e,
    experiment2 x,
    param p,
    attr a
where TRUE
    and e.experiment_id = x.experiment_id
    and x.experiment_attrs @> array[37]
    and x.experiment_attrs @> array[a.attr_id]
    and e.experiment_parameters @> array[p.id]
group by x.*, e.*


select 
    dataset_name, model_input_shape, model_output_units,
    count(distinct experiment_uid) num_exp,
    count(1) num_run
FROM (
select 
    e.experiment_uid,
    a_dataset_name.value_str dataset_name,
    a_model_input_shape.value_json model_input_shape,
    a_model_output_units.value_int model_output_units
from
    experiment2 e,
    run2 r,
    attr a_dataset_name,
    attr a_model_input_shape,
    attr a_model_output_units,
    attr regularization
where
    e.experiment_attrs && (
        select array_agg(attr_id) from attr
            where kind = 'dataset_name'
    )
    and r.experiment_uid = e.experiment_uid
    and a_dataset_name.kind = 'dataset_name'
    and e.experiment_attrs @> ARRAY[a_dataset_name.attr_id]
    and a_model_input_shape.kind = 'model_input_shape'
    and ARRAY[a_model_input_shape.attr_id] <@ e.experiment_attrs 
    and a_model_output_units.kind = 'model_output_units'
    and ARRAY[a_model_output_units.attr_id] <@ e.experiment_attrs
) x
group by dataset_name, model_input_shape, model_output_units;

select * from attr where kind like 'model_output%';
    

select distinct error_message from experiment_migration m where is_valid and migrated and error_message is not null 
and error_message NOT LIKE 'failed on Could not find%' and error_message NOT LIKE 'wrong number%wide_first_%' AND error_message NOT LIKE 'wrong number%poker%'
limit 1000;



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

---uuid.UUID(hashlib.md5(b'{3,4,5,8,9,10,12,13,18,19,21,34,40,270,670,1377,1378,1387,1402}').hexdigest())

CREATE TABLE attr
(
    attr_id serial NOT NULL,
    value_type smallint NOT NULL,
    kind text NOT NULL,
    value_bool boolean,
    value_int bigint,
    value_float double precision,
    value_str text,
    digest uuid,
    value_json jsonb,
    PRIMARY KEY (attr_id)
);
ALTER TABLE attr SET (fillfactor = 100);

ALTER TABLE attr
  ADD CONSTRAINT parameter_proper_format
  CHECK (
    (value_type = 0 
        AND value_bool IS NULL
        AND value_int IS NULL
        AND value_float IS NULL
        AND value_str IS NULL
        AND value_json IS NULL
        AND digest IS NULL
    )
    OR
    (value_type = 1
        AND value_bool IS NOT NULL
        AND value_int IS NULL
        AND value_float IS NULL
        AND value_str IS NULL
        AND value_json IS NULL
        AND digest IS NULL
    )
    OR
    (value_type = 2 
        AND value_bool IS NULL
        AND value_int IS NOT NULL
        AND value_float IS NULL
        AND value_str IS NULL
        AND value_json IS NULL
        AND digest IS NULL
    )
    OR
    (value_type = 3
        AND value_bool IS NULL
        AND value_int IS NULL
        AND value_float IS NOT NULL
        AND value_str IS NULL
        AND value_json IS NULL
        AND digest IS NULL
    )
    OR
    (value_type = 4
        AND value_bool IS NULL
        AND value_int IS NULL
        AND value_float IS NULL
        AND value_str IS NOT NULL
        AND value_json IS NULL
        AND digest IS NULL
    )
    OR
    (value_type = 5
        AND value_bool IS NULL
        AND value_int IS NULL
        AND value_float IS NULL
        AND value_str IS NULL
        AND value_json IS NOT NULL
        AND digest IS NOT NULL
    )
  );
  
CREATE UNIQUE INDEX ON attr USING btree (kind) INCLUDE (attr_id) WHERE value_type = 0;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_bool) INCLUDE (attr_id) WHERE value_type = 1;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_int) INCLUDE (attr_id) WHERE value_type = 2;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_float) INCLUDE (attr_id) WHERE value_type = 3;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_str) INCLUDE (attr_id) WHERE value_type = 4;
CREATE UNIQUE INDEX ON attr USING btree (kind, digest) WHERE value_type = 5;
-- CREATE UNIQUE INDEX ON attr USING btree (kind, value_json) WHERE value_type = 5;

CREATE INDEX ON attr USING btree (kind);
CREATE INDEX ON attr USING gin (value_json) WHERE value_type = 5;

CREATE TABLE experiment2
(
    experiment_uid uuid NOT NULL,
    experiment_id integer,
    experiment_attrs integer[] NOT NULL,
    PRIMARY KEY (experiment_uid)
);

ALTER TABLE experiment2 SET (fillfactor = 100);
ALTER TABLE experiment2 SET (parallel_workers = 16);
ALTER TABLE experiment2 ALTER COLUMN experiment_attrs SET storage PLAIN;

CREATE INDEX on experiment2 USING btree (experiment_id) WHERE experiment_id IS NOT NULL;
CREATE INDEX ON experiment2 USING gin (experiment_attrs);

CREATE TABLE run2
(
    experiment_uid uuid,

    run_timestamp timestamp DEFAULT CURRENT_TIMESTAMP,
    
    run_id uuid NOT NULL,
    job_id uuid,

    seed bigint,
    slurm_job_id bigint,

    task_version smallint,
    num_nodes smallint,
    num_cpus smallint,
    num_gpus smallint,
    gpu_memory integer,

    host_name text,
    batch text,
    
    run_data jsonb,
    run_history bytea,
    
    PRIMARY KEY (run_id)
);


ALTER TABLE run2 ALTER COLUMN run_history SET storage EXTERNAL;

ALTER TABLE run2 SET (fillfactor = 100);
ALTER TABLE run2 SET (parallel_workers = 16);

CREATE INDEX ON run2 USING btree (experiment_uid);

CREATE INDEX ON run2 USING btree (run_timestamp) INCLUDE (experiment_uid);
CREATE INDEX ON run2 USING btree (experiment_uid, run_timestamp);

CREATE INDEX ON run2 USING btree (job_id) WHERE job_id IS NOT NULL;
CREATE INDEX ON run2 USING btree (slurm_job_id) WHERE slurm_job_id IS NOT NULL;
CREATE INDEX ON run2 USING btree (task_version) WHERE task_version IS NOT NULL;
CREATE INDEX ON run2 USING btree (num_nodes) WHERE num_nodes IS NOT NULL;
CREATE INDEX ON run2 USING btree (num_cpus) WHERE num_cpus IS NOT NULL;
CREATE INDEX ON run2 USING btree (num_gpus) WHERE num_gpus IS NOT NULL;
CREATE INDEX ON run2 USING btree (host_name) WHERE host_name IS NOT NULL;
CREATE INDEX ON run2 USING btree (batch) WHERE batch IS NOT NULL;

CREATE INDEX ON run2 USING gin (run_data);
CREATE INDEX ON run2 USING hash (experiment_uid);

CREATE TABLE experiment_summary
(
    experiment_uid uuid,
    last_updated timestamp DEFAULT CURRENT_TIMESTAMP,
    core_data bytea,
    extended_data bytea,
    PRIMARY KEY (experiment_uid)
);

CREATE INDEX ON experiment_summary USING btree (experiment_uid) INCLUDE (last_updated);
CREATE INDEX ON experiment_summary USING btree (last_updated);
CREATE INDEX ON experiment_summary USING btree (last_updated, experiment_uid);
CREATE INDEX ON experiment_summary USING hash (last_updated);


CREATE TABLE experiment_summary_progress
(
    last_updated timestamp
);