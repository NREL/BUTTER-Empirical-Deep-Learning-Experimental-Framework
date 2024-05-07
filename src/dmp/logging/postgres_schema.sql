CREATE TABLE IF NOT EXISTS parameter
(
    id SMALLINT NOT NULL DEFAULT nextval('parameter_id_seq'::regclass),
    kind TEXT COLLATE pg_catalog."default" NOT NULL,
    bool_value BOOLEAN,
    real_value REAL,
    integer_value BIGINT,
    string_value TEXT,
    CONSTRAINT PRIMARY KEY (id)
);

CREATE SEQUENCE IF NOT EXISTS parameter_id_seq
    INCREMENT 1
    START -32699
    MINVALUE -32768
    MAXVALUE 32767
    CACHE 1
    OWNED BY parameter.id;

CREATE INDEX ON parameter USING btree (kind);

CREATE INDEX ON parameter USING btree (kind) WHERE bool_value IS NULL AND real_value IS NULL AND integer_value IS NULL and string_value IS NULL;
CREATE INDEX ON parameter USING btree (kind, bool_value) WHERE bool_value IS NOT NULL;
CREATE INDEX ON parameter USING btree (kind, real_value) WHERE real_value IS NOT NULL;
CREATE INDEX ON parameter USING btree (kind, integer_value) WHERE integer_value IS NOT NULL;
CREATE INDEX ON parameter USING btree (kind, string_value) WHERE string_value IS NOT NULL;

CREATE TABLE IF NOT EXISTS sweep
(
    id SMALLINT NOT NULL DEFAULT nextval('sweep_id_seq'::regclass),
    name TEXT,
    CONSTRAINT PRIMARY KEY (id)
);

CREATE INDEX ON sweep (name);


CREATE SEQUENCE IF NOT EXISTS sweep_id_seq
    INCREMENT 1
    START 0
    MINVALUE -32768
    MAXVALUE 32767
    CACHE 1
    OWNED BY sweep.id;

CREATE TABLE IF NOT EXISTS experiment
(
    experiment_id BIGSERIAL NOT NULL,
    experiment_parameters SMALLINT[] NOT NULL,
    sweeps SMALLINT[],
    experiment_data jsonb not null,
    model_computed_network_structure jsonb,
    relative_size_error REAL,
    CONSTRAINT PRIMARY KEY (experiment_parameters)
);

CREATE INDEX ON experiment (experiment_id);
CREATE INDEX ON experiment USING GIN (experiment_parameters);
CREATE INDEX ON experiment USING GIN (experiment_data);
CREATE INDEX ON experiment USING GIN (sweeps);


CREATE TABLE IF NOT EXISTS run
(
    experiment_id BIGINT NOT NULL,
    slurm_job_id BIGINT,

    job_id uuid NOT NULL,
    run_id uuid NOT NULL,

    start_time timestamp WITH TIME ZONE,
    end_time timestamp WITH TIME ZONE,
    record_time timestamp WITH TIME ZONE DEFAULT NOW(),

    duration_in_ms INTEGER,

    task_version SMALLINT,
    queue_id SMALLINT,
    save_every_epochs SMALLINT,
    num_gpus SMALLINT,
    num_nodes SMALLINT,
    num_cpus SMALLINT,
    gpu_memory INTEGER,

    batch TEXT,
    system_name TEXT,
    host_name TEXT,
    tensorflow_strategy TEXT,

    run_data jsonb NOT NULL,
    run_history bytes[] NOT NULL,

    CONSTRAINT PRIMARY KEY (run_id)
);

ALTER TABLE IF EXISTS public.run_
    ALTER COLUMN run_history SET STORAGE EXTERNAL;

create index on run (experiment_id);
create index on run USING GIN (run_data);


create index on run (system_name, start_time);
create index on run (system_name, num_gpus, num_cpus, start_time);
create index on run (system_name, num_cpus);

create index on run using hash (job_id);



------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS experiment_
(
    experiment_id serial NOT NULL,
    experiment_parameters SMALLINT[] NOT NULL PRIMARY KEY,
    num_free_parameters BIGINT,
    widths SMALLINT[],
    network_structure jsonb
);

alter table experiment_ alter column network_structure set storage EXTENDED;
alter table experiment_ alter column widths set storage EXTENDED;

create unique index on experiment_ (experiment_id);
create index on experiment_ using gin (experiment_parameters);


CREATE TABLE IF NOT EXISTS run_
(
    experiment_id INTEGER,
    record_timestamp INTEGER DEFAULT ((date_part('epoch'::TEXT, CURRENT_TIMESTAMP) - (1600000000)::double precision))::INTEGER,
    run_id uuid NOT NULL PRIMARY KEY,
    job_id uuid,
    run_parameters SMALLINT[] NOT NULL,
    test_loss real[],
    train_loss real[],
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
    platform TEXT,
    git_hash TEXT,
    hostname TEXT,
    slurm_job_id TEXT,
    seed BIGINT,
    save_every_epochs SMALLINT
);

alter table run_ alter column test_loss set storage EXTERNAL;
alter table run_ alter column train_loss set storage EXTERNAL;
alter table run_ alter column test_accuracy set storage EXTERNAL;
alter table run_ alter column train_accuracy set storage EXTERNAL;
alter table run_ alter column test_mean_squared_error set storage EXTERNAL;
alter table run_ alter column train_mean_squared_error set storage EXTERNAL;
alter table run_ alter column test_mean_absolute_error set storage EXTERNAL;
alter table run_ alter column train_mean_absolute_error set storage EXTERNAL;
alter table run_ alter column test_root_mean_squared_error set storage EXTERNAL;
alter table run_ alter column train_root_mean_squared_error set storage EXTERNAL;
alter table run_ alter column test_mean_squared_logarithmic_error set storage EXTERNAL;
alter table run_ alter column train_mean_squared_logarithmic_error set storage EXTERNAL;
alter table run_ alter column test_hinge set storage EXTERNAL;
alter table run_ alter column train_hinge set storage EXTERNAL;
alter table run_ alter column test_squared_hinge set storage EXTERNAL;
alter table run_ alter column train_squared_hinge set storage EXTERNAL;
alter table run_ alter column test_cosine_similarity set storage EXTERNAL;
alter table run_ alter column train_cosine_similarity set storage EXTERNAL;
alter table run_ alter column test_kullback_leibler_divergence set storage EXTERNAL;
alter table run_ alter column train_kullback_leibler_divergence set storage EXTERNAL;


-- create index on run_ using hash(job_id);
create index on run_ using hash (experiment_id);
create index on run_ (record_timestamp, experiment_id);
create index on run_ using gin (run_parameters);



CREATE TABLE IF NOT EXISTS parameter_
(
    id smallserial NOT NULL primary key,
    bool_value BOOLEAN,
    real_value real,
    integer_value BIGINT,
    string_value TEXT COLLATE pg_catalog."default",
    kind TEXT COLLATE pg_catalog."default" NOT NULL
);

CREATE UNIQUE INDEX on parameter_ (kind, bool_value) include (id) where bool_value is not null;
CREATE UNIQUE INDEX on parameter_ (kind, real_value) include (id) where real_value is not null;
CREATE UNIQUE INDEX on parameter_ (kind, integer_value) include (id) where integer_value is not null;
CREATE UNIQUE INDEX on parameter_ (kind, string_value) include (id) where string_value is not null;
CREATE UNIQUE INDEX on parameter_ (kind) include (id) where bool_value is null and real_value is null and integer_value is null and string_value is null;



CREATE TABLE IF NOT EXISTS experiment_summary_
(
    experiment_id INTEGER NOT NULL primary key,
    update_timestamp INTEGER NOT NULL DEFAULT ((date_part('epoch'::TEXT, CURRENT_TIMESTAMP) - (1600000000)::double precision))::INTEGER,
    experiment_parameters SMALLINT[] NOT NULL,
    num_runs SMALLINT NOT NULL,
    num_free_parameters INTEGER,
    num SMALLINT[],
    test_loss_num_finite SMALLINT[],
    test_loss_avg real[],
    test_loss_stddev real[],
    test_loss_min real[],
    test_loss_max real[],
    test_loss_percentile real[],
    train_loss_num_finite SMALLINT[],
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
    train_kullback_leibler_divergence_stddev real[],
    network_structure jsonb,
    widths INTEGER[]
);


CREATE INDEX ON experiment_summary_ USING hash (experiment_id);
CREATE INDEX ON experiment_summary_ USING gin (experiment_parameters);
create index on experiment_summary_ (update_timestamp, experiment_id);




update experiment_ e set
    "size" = size_.integer_value,
    "relative_size_error" = (abs( (size_.integer_value - e.num_free_parameters) / (size_.integer_value)::float))::real
FROM
parameter_ as size_
WHERE
(e.size is NULL OR e.relative_size_error is NULL) and
e.experiment_parameters @> array[size_.id] and size_.kind = 'size';

update experiment_summary_ s
set
    "size" = e.size,
    relative_size_error = e.relative_size_error
from
    experiment_ e
where
    s.experiment_id = e.experiment_id and
    e.size is not null and
    e.relative_size_error is not null and
    (s.size is null or s.relative_size_error is null);


vacuum experiment_summary_;









with b as (
select s.id
from
    job_data d,
    job_status s
where
    d.id = s.id and
    s.queue = 1 and
    (s.status = 0 or s.status = 3) and
    exists (
        select
            size_.integer_value size,
            shape_.string_value shape,
            depth_.integer_value depth,
            dataset_.string_value dataset,
            relative_size_error
        from
            experiment_ e,
            parameter_ size_,
            parameter_ shape_,
            parameter_ depth_,
            parameter_ dataset_
        where
            e.relative_size_error > .2 and
            e.experiment_parameters @> array[size_.id] and size_.kind = 'size' and size_.integer_value = (d.command->'size')::BIGINT and
            e.experiment_parameters @> array[shape_.id] and shape_.kind = 'shape' and shape_.string_value = d.command->>'shape' and
            e.experiment_parameters @> array[depth_.id] and depth_.kind = 'depth' and depth_.integer_value = (d.command->'depth')::BIGINT and
            e.experiment_parameters @> array[dataset_.id] and dataset_.kind = 'dataset' and  dataset_.string_value = d.command->>'dataset'
        )
),
d1 as (
delete from job_status s using b where s.id = b.id
)
delete from job_data s using b where s.id = b.id
;