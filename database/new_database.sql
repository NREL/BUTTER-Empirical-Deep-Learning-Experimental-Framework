drop table run2;
drop table experiment2;
drop table parameter2;

truncate table run2;
truncate table experiment2;
truncate table parameter2;

CREATE TABLE parameter2
(
    value_int bigint,
    value_float double precision,
    parameter_id serial NOT NULL,
    value_type smallint NOT NULL,
    value_bool boolean,
    value_json jsonb,
    value_str text,
    kind text NOT NULL,
    PRIMARY KEY (parameter_id)
);

ALTER TABLE parameter2
  ADD CONSTRAINT parameter_constrain_proper_type
  CHECK (
    (value_type = 0 
        AND value_bool IS NULL
        AND value_int IS NULL
        AND value_float IS NULL
        AND value_str IS NULL
        AND value_json IS NULL
    )
    OR
    (value_type = 1
        AND value_bool IS NOT NULL
        AND value_int IS NULL
        AND value_float IS NULL
        AND value_str IS NULL
        AND value_json IS NULL
    )
    OR
    (value_type = 2 
        AND value_bool IS NULL
        AND value_int IS NOT NULL
        AND value_float IS NULL
        AND value_str IS NULL
        AND value_json IS NULL
    )
    OR
    (value_type = 3
        AND value_bool IS NULL
        AND value_int IS NULL
        AND value_float IS NOT NULL
        AND value_str IS NULL
        AND value_json IS NULL
    )
    OR
    (value_type = 4
        AND value_bool IS NULL
        AND value_int IS NULL
        AND value_float IS NULL
        AND value_str IS NOT NULL
        AND value_json IS NULL
    )
    OR
    (value_type = 5
        AND value_bool IS NULL
        AND value_int IS NULL
        AND value_float IS NULL
        AND value_str IS NULL
        AND value_json IS NOT NULL
    )
  );
  
CREATE UNIQUE INDEX ON parameter2 USING btree (kind) INCLUDE (parameter_id) WHERE value_type = 0;
CREATE UNIQUE INDEX ON parameter2 USING btree (kind, value_bool) INCLUDE (parameter_id) WHERE value_type = 1;
CREATE UNIQUE INDEX ON parameter2 USING btree (kind, value_int) INCLUDE (parameter_id) WHERE value_type = 2;
CREATE UNIQUE INDEX ON parameter2 USING btree (kind, value_float) INCLUDE (parameter_id) WHERE value_type = 3;
CREATE UNIQUE INDEX ON parameter2 USING btree (kind, value_str) INCLUDE (parameter_id) WHERE value_type = 4;
CREATE UNIQUE INDEX ON parameter2 USING btree (kind, value_json) WHERE value_type = 5;

CREATE INDEX ON parameter2 USING btree (kind);
CREATE INDEX ON parameter2 USING gin (value_json) WHERE value_type = 5;


CREATE TABLE experiment2
(
    experiment_id serial NOT NULL,
    experiment_parameters integer[] NOT NULL,
    experiment_attributes integer[],
    experiment_data jsonb,
    model_structure bytea,
    PRIMARY KEY (experiment_id),
    UNIQUE (experiment_parameters)
);

CREATE INDEX ON experiment2 USING gin (experiment_parameters);
CREATE INDEX ON experiment2 USING gin (experiment_attributes);
CREATE INDEX ON experiment2 USING gin (experiment_data);

CREATE TABLE run2
(
    experiment_id integer,
    
    task_version smallint,
    num_nodes smallint,

    record_timestamp timestamp DEFAULT CURRENT_TIMESTAMP,
    run_attributes integer[] NOT NULL,
    
    slurm_job_id bigint,
    job_id uuid NOT NULL,
    run_id uuid NOT NULL,

    num_cpus smallint,
    num_gpus smallint,
    gpu_memory integer,
    
    seed bigint,
    host_name text,
    
    run_data jsonb,
    run_history bytea,
    PRIMARY KEY (run_id)
);

CREATE INDEX ON run2 USING btree (experiment_id);

CREATE INDEX ON run2 USING btree (record_timestamp) INCLUDE (experiment_id);
CREATE INDEX ON run2 USING btree (experiment_id, record_timestamp);

CREATE INDEX ON run2 USING btree (job_id);
CREATE INDEX ON run2 USING btree (slurm_job_id);

CREATE INDEX ON run2 USING gin (run_parameters);
CREATE INDEX ON run2 USING gin (run_data);



