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

SELECT *
FROM pg_stat_activity 
WHERE usename = 'dmpappsops'
ORDER BY state, query_start desc;

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

CREATE TABLE experiment_migration (
    experiment_id integer,
    migrated boolean,
    is_valid boolean,
    PRIMARY KEY (experiment_id)
);

CREATE INDEX ON experiment_migration USING btree(experiment_id) WHERE NOT migrated AND is_valid;

INSERT INTO experiment_migration (experiment_id, migrated, is_valid)
SELECT
    experiment_id,
    FALSE,
    TRUE
FROM
    experiment_ e
WHERE EXISTS (select 1 from run_ r where r.experiment_id = e.experiment_id);

UPDATE experiment_migration set migrated = FALSE, error_message =NULL WHERE migrated and error_message IS NOT NULL;

update experiment_migration m set 
    is_valid = True,
    migrated = False,
    error_message = NULL
    WHERE EXISTS (select 1 from run_ r where r.experiment_id = m.experiment_id)
    AND NOT EXISTS (select 1 from experiment e where e.old_experiment_id = m.experiment_id)
    and error_message NOT LIKE 'failed on Could not find%' 
    and error_message NOT LIKE 'wrong number%';

update experiment_migration m
    set migrated = FALSE
WHERE 
    migrated
    AND error_message is null 
    AND (
        select count(1) from run r, experiment e 
        where e.old_experiment_id = m.experiment_id 
        and r.experiment_id = e.experiment_id) <> 
        (select count(1) from run_ r where r.experiment_id = m.experiment_id);


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
    experiment e,
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
    count(distinct experiment_id) num_exp,
    count(1) num_run
FROM (
select 
    e.experiment_id,
    a_dataset_name.value_str dataset_name,
    a_model_input_shape.value_json model_input_shape,
    a_model_output_units.value_int model_output_units
from
    experiment e,
    run r,
    attr a_dataset_name,
    attr a_model_input_shape,
    attr a_model_output_units
where
    e.experiment_attrs && (
        select array_agg(attr_id) from attr
            where kind = 'dataset_name'
    )
    and r.experiment_id = e.experiment_id
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
        from attr left join experiment on (experiment_attributes @> array[attribute_id])
        group by kind, value_type, v
    ) ac
    group by kind, value_type
    order by kind, value_type
) x
order by compression_ratio asc;

select * from attr where kind like '%regul%';

select count(1) from experiment where experiment_attrs @> array[37];


select 
    e.*,
    x.*,
    json_object_agg(a.kind, coalesce(a.value_bool, a.value_int, a.value_float, a.value_str, a.value_json)) xattrib,
    json_object_agg(p.kind, coalesce(p.bool_value, p.int_value, p.real_value, p.string_value)) eattrib
from
    experiment_ e,
    experiment x,
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
    count(distinct experiment_id) num_exp,
    count(1) num_run
FROM (
select 
    e.experiment_id,
    a_dataset_name.value_str dataset_name,
    a_model_input_shape.value_json model_input_shape,
    a_model_output_units.value_int model_output_units
from
    experiment e,
    run r,
    attr a_dataset_name,
    attr a_model_input_shape,
    attr a_model_output_units,
    attr regularization
where
    e.experiment_attrs && (
        select array_agg(attr_id) from attr
            where kind = 'dataset_name'
    )
    and r.experiment_id = e.experiment_id
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

CREATE TABLE experiment
(
    experiment_id uuid NOT NULL,
    old_experiment_id integer,
    experiment_attrs integer[] NOT NULL,
    experiment_properties integer[],
    PRIMARY KEY (experiment_id)
);

ALTER TABLE experiment SET (fillfactor = 100);
ALTER TABLE experiment SET (parallel_workers = 16);
ALTER TABLE experiment ALTER COLUMN experiment_attrs SET storage PLAIN;

CREATE INDEX on experiment USING btree (experiment_id) WHERE experiment_id IS NOT NULL;
CREATE INDEX ON experiment USING gin (experiment_attrs, experiment_properties);
CREATE INDEX ON experiment USING btree (old_experiment_id) INCLUDE (experiment_id) WHERE old_experiment_id IS NOT NULL;

CREATE TABLE run
(
    experiment_id uuid,

    run_timestamp timestamp WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
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
    run_extended_history bytea,
    
    PRIMARY KEY (run_id)
);

ALTER TABLE run SET (toast_tuple_target = 128);

ALTER TABLE run ALTER COLUMN run_history SET storage EXTERNAL;
ALTER TABLE run ALTER COLUMN run_extended_history SET storage EXTERNAL;

ALTER TABLE run SET (fillfactor = 100);
ALTER TABLE run SET (parallel_workers = 16);

CREATE INDEX ON run USING btree (experiment_id);
CREATE INDEX ON run USING hash (experiment_id);

CREATE INDEX ON run USING btree (experiment_id, run_timestamp);
CREATE INDEX ON run USING btree (run_timestamp DESC, experiment_id);

CREATE INDEX ON run USING btree (job_id) WHERE job_id IS NOT NULL;
CREATE INDEX ON run USING btree (slurm_job_id) WHERE slurm_job_id IS NOT NULL;
CREATE INDEX ON run USING btree (task_version) WHERE task_version IS NOT NULL;
CREATE INDEX ON run USING btree (num_nodes) WHERE num_nodes IS NOT NULL;
CREATE INDEX ON run USING btree (num_cpus) WHERE num_cpus IS NOT NULL;
CREATE INDEX ON run USING btree (num_gpus) WHERE num_gpus IS NOT NULL;
CREATE INDEX ON run USING btree (host_name) WHERE host_name IS NOT NULL;
CREATE INDEX ON run USING btree (batch) WHERE batch IS NOT NULL;

CREATE INDEX ON run USING gin (run_data);

CREATE TABLE experiment_summary
(
    experiment_id uuid,
    last_updated timestamp,
    most_recent_run timestamp,
    by_epoch bytea,
    by_loss bytea,
    by_progress bytea,
    epoch_subset bytea,
    PRIMARY KEY (experiment_id)
);

alter table experiment_summary alter column claim_time set default '1960-01-01'::timestamp;

ALTER TABLE experiment_summary ALTER COLUMN by_epoch SET STORAGE EXTERNAL;
ALTER TABLE experiment_summary ALTER COLUMN by_loss SET STORAGE EXTERNAL;
ALTER TABLE experiment_summary ALTER COLUMN by_progress SET STORAGE EXTERNAL;
ALTER TABLE experiment_summary ALTER COLUMN epoch_subset SET STORAGE EXTERNAL;

CREATE INDEX ON experiment_summary USING hash(experiment_id);
-- CREATE INDEX ON experiment_summary USING btree (last_updated DESC, experiment_id);
-- CREATE INDEX ON experiment_summary (experiment_id) INCLUDE (last_updated, most_recent_run);
-- CREATE INDEX ON experiment_summary USING btree (experiment_id, last_updated);
-- CREATE INDEX ON experiment_summary USING btree (experiment_id, most_recent_run);


--- to check run data sizes:
SELECT 
    pg_size_pretty(avg(length(run_history))) avg_run_history_size,
    pg_size_pretty(avg(length(run_extended_history))) avg_run_extended_history_size
FROM run;

SELECT 
    pg_size_pretty(avg(length(by_epoch))) avg_size_by_epoch,
    pg_size_pretty(avg(length(by_loss))) avg_size_by_loss,
    pg_size_pretty(avg(length(by_progress))) avg_size_by_progress,
    pg_size_pretty(avg(length(epoch_subset))) avg_size_epoch_subset
FROM experiment_summary;

--- TO REQUEUE lost experiment_summaries:

insert into experiment_summary (experiment_id)
select experiment_id from experiment;

UPDATE experiment_summary set 
    last_updated = '1960-01-01'::timestamp
WHERE
    by_epoch IS NULL
    OR
    (
    SELECT MAX(run_timestamp) FROM run WHERE run.experiment_id = experiment_summary.experiment_id
    ) > LEAST(experiment_summary.last_updated, experiment_summary.most_recent_run);

select
    *,
    (100.0 * incomplete / total) pct_incomplete,
    100.0 - (100.0 * incomplete / total) pct_complete
FROM
    (select 
        sum((by_epoch is null)::integer) incomplete, 
        sum((most_recent_run IS NOT NULL and by_epoch IS NULL)::integer) pending,
        count(1) total,
        pg_size_pretty(avg(length(by_epoch))) avg_size_by_epoch,
        pg_size_pretty(avg(length(by_loss))) avg_size_by_loss,
        pg_size_pretty(avg(length(by_progress))) avg_size_by_progress,
        pg_size_pretty(avg(length(epoch_subset))) avg_size_epoch_subset
     from experiment_summary
     ) x
     ;

--- TO MOVE attrs to properties

WITH property_attrs as (
    select attr_id
    from attr where kind like '%sweep%' or kind = 'butter'
),
exp_target as (
    select 
        e.experiment_id src_id,  
        (md5(x.experiment_attrs::text))::uuid dst_id, 
        x.experiment_attrs,
        y.experiment_properties,
        old_experiment_id
    from 
        experiment e,
        lateral (select array_agg(attr_id) experiment_attrs from (
            select attr_id
            from
                unnest(e.experiment_attrs) a(attr_id)
            where attr_id not in (select attr_id from property_attrs)
            order by attr_id) x
        ) x,
        lateral (
            select array_agg(attr_id) experiment_properties from (
            select attr_id
            from
                unnest(e.experiment_attrs) a(attr_id)
            where attr_id in (select attr_id from property_attrs)
            order by attr_id) x
        ) y
    WHERE e.experiment_attrs && (select array_agg(attr_id) from property_attrs)
    order by dst_id
        
),
dst_map as (
    select distinct on (dst_id) *
    FROM
    (
    SELECT
        dst_id,
        experiment_attrs,
        first_value(experiment_properties) over (partition by dst_id ORDER BY greatest(array_length(experiment_properties, 1)) DESC) experiment_properties,
        first_value(old_experiment_id) over (partition by dst_id ORDER BY old_experiment_id ASC) old_experiment_id
     FROM (
         SELECT 
            dst_id,
            experiment_attrs,
            experiment_properties,
            old_experiment_id
         FROM exp_target
         UNION ALL
         (
             SELECT 
                experiment_id dst_id,
                experiment_attrs,
                experiment_properties,
                old_experiment_id
             FROM
                experiment e
             WHERE
                e.experiment_id in (SELECT dst_id from exp_target)
         )
     ) x
    ) x
    ORDER BY dst_id
),
exp_map as (
    select 
        src_id,
        dst_map.*
    from
        dst_map inner join exp_target using (dst_id)
    order by src_id
),
exp_delete as (
    DELETE FROM experiment e
    USING exp_map m
    WHERE m.src_id = e.experiment_id
),
exp_update as (
    INSERT INTO experiment (
        experiment_id,
        experiment_attrs,
        experiment_properties,
        old_experiment_id
        )
    SELECT 
        dst_id experiment_id,
        experiment_attrs,
        experiment_properties,
        old_experiment_id
    FROM
        dst_map m
    ON CONFLICT (experiment_id) DO UPDATE SET
        experiment_properties = EXCLUDED.experiment_properties,
        old_experiment_id = EXCLUDED.old_experiment_id
)
update run r set
    experiment_id = m.dst_id
from 
    exp_map m
where
    r.experiment_id = m.src_id
;
