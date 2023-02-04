drop table run;
drop table experiment;
drop table parameter2;

truncate table run;
truncate table experiment;
truncate table parameter2;

ALTER SEQUENCE parameter2_parameter_id_seq RESTART WITH 1;

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

ALTER TABLE parameter2 SET (fillfactor = 100);

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


CREATE TABLE experiment
(
    experiment_id serial NOT NULL,
    experiment_parameters integer[] NOT NULL,
    experiment_attrs integer[],
    experiment_data jsonb,
    model_structure jsonb,
    PRIMARY KEY (experiment_id),
    UNIQUE (experiment_parameters)
);

ALTER TABLE experiment SET (fillfactor = 100);
ALTER TABLE experiment SET (parallel_workers = 16);

ALTER TABLE experiment ALTER COLUMN experiment_parameters SET storage PLAIN;
ALTER TABLE experiment ALTER COLUMN experiment_attrs SET storage PLAIN;

CREATE INDEX ON experiment USING gin (experiment_parameters);
CREATE INDEX ON experiment USING gin (experiment_attrs);
CREATE INDEX ON experiment USING gin (experiment_data);

CREATE TABLE run
(
    experiment_id integer NOT NULL,
    
    task_version smallint,
    num_nodes smallint,

    record_timestamp timestamp DEFAULT CURRENT_TIMESTAMP,
        
    slurm_job_id bigint,
    job_id uuid NOT NULL,
    run_id uuid NOT NULL,

    num_cpus smallint,
    num_gpus smallint,
    gpu_memory integer,
    
    seed bigint,
    host_name text,
    
    run_attributes integer[] NOT NULL,
    run_data jsonb,
    run_history bytea,
    PRIMARY KEY (run_id)
);

ALTER TABLE run ALTER COLUMN run_attributes SET storage PLAIN;
ALTER TABLE run ALTER COLUMN run_history SET storage EXTERNAL;

ALTER TABLE run SET (fillfactor = 100);
ALTER TABLE run SET (parallel_workers = 16);

CREATE INDEX ON run USING btree (experiment_id);

CREATE INDEX ON run USING btree (record_timestamp) INCLUDE (experiment_id);
CREATE INDEX ON run USING btree (experiment_id, record_timestamp);

CREATE INDEX ON run USING btree (job_id);
CREATE INDEX ON run USING btree (slurm_job_id);

CREATE INDEX ON run USING gin (run_parameters);
CREATE INDEX ON run USING gin (run_data);



