
---uuid.UUID(hashlib.md5(b'{3,4,5,8,9,10,12,13,18,19,21,34,40,270,670,1377,1378,1387,1402}').hexdigest())

CREATE TABLE attr
(
    attribute_id serial NOT NULL,
    value_type smallint NOT NULL,
    kind text NOT NULL,
    value_bool boolean,
    value_int bigint,
    value_float double precision,
    value_str text,
    value_json jsonb,
    PRIMARY KEY (attribute_id)
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
  
CREATE UNIQUE INDEX ON attr USING btree (kind) INCLUDE (attribute_id) WHERE value_type = 0;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_bool) INCLUDE (attribute_id) WHERE value_type = 1;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_int) INCLUDE (attribute_id) WHERE value_type = 2;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_float) INCLUDE (attribute_id) WHERE value_type = 3;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_str) INCLUDE (attribute_id) WHERE value_type = 4;
CREATE UNIQUE INDEX ON attr USING btree (kind, value_json) WHERE value_type = 5;

CREATE INDEX ON attr USING btree (kind);
CREATE INDEX ON attr USING gin (value_json) WHERE value_type = 5;

CREATE TABLE experiment2
(
    experiment_uid uuid NOT NULL,
    experiment_id integer,
    experiment_attributes integer[] NOT NULL,
    PRIMARY KEY (experiment_uid)
);

ALTER TABLE experiment2 SET (fillfactor = 100);
ALTER TABLE experiment2 SET (parallel_workers = 16);
ALTER TABLE experiment2 ALTER COLUMN experiment_attributes SET storage PLAIN;

CREATE INDEX on experiment2 USING btree (experiment_id) WHERE experiment_id IS NOT NULL;
CREATE INDEX ON experiment2 USING gin (experiment_attributes);

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

CREATE INDEX ON run2 USING btree (record_timestamp) INCLUDE (experiment_uid);
CREATE INDEX ON run2 USING btree (experiment_uid, record_timestamp);

CREATE INDEX ON run2 USING btree (job_id);
CREATE INDEX ON run2 USING btree (slurm_job_id);
CREATE INDEX ON run2 USING btree (task_version);
CREATE INDEX ON run2 USING btree (num_nodes);
CREATE INDEX ON run2 USING btree (num_cpus);
CREATE INDEX ON run2 USING btree (num_gpus);
CREATE INDEX ON run2 USING btree (host_name);
CREATE INDEX ON run2 USING btree (batch);

CREATE INDEX ON run2 USING gin (run_data);


