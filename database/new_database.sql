+ start loading, start training, etc


CREATE TABLE parameter
(
    id serial NOT NULL,
    value_type smallint NOT NULL,
    kind text NOT NULL,
    bool_value boolean,
    integer_value bigint,
    float_value double precision,
    string_value text,
    PRIMARY KEY (id)
);

CREATE INDEX ON parameter USING btree (kind);
CREATE INDEX ON parameter USING btree (kind, id) WHERE value_type = 0;
CREATE INDEX ON parameter USING btree (kind, bool_value, id) WHERE value_type = 1;
CREATE INDEX ON parameter USING btree (kind, integer_value, id) WHERE value_type = 2;
CREATE INDEX ON parameter USING btree (kind, float_value, id) WHERE value_type = 3;
CREATE INDEX ON parameter USING btree (kind, string_value, id) WHERE value_type = 4;


CREATE TABLE experiment
(
    experiment_id serial NOT NULL,
    experiment_parameters integer[] NOT NULL,
    experiment_attributes integer[],
    model_structure bytea,
    PRIMARY KEY (experiment_parameters)
);

CREATE INDEX ON experiment USING gin (experiment_parameters);
CREATE INDEX ON experiment USING gin (experiment_attributes);


CREATE TABLE run
(
    experiment_id integer,
    
    task_version smallint,
    num_nodes smallint,

    record_timestamp timestamp DEFAULT CURRENT_TIMESTAMP,
    run_parameters integer[] NOT NULL,
    
    slurm_job_id bigint,
    job_id uuid NOT NULL,
    run_id uuid NOT NULL,

    num_cpus smallint,
    num_gpus smallint,
    gpu_memory integer,
    
    seed bigint,
    host_name text,
    
    run_history bytea,
    PRIMARY KEY (run_id)
);

CREATE INDEX ON run USING btree (experiment_id);
CREATE INDEX ON run USING btree (record_timestamp, experiment_id);
CREATE INDEX ON run USING gin (run_parameters);
CREATE INDEX ON run USING btree (job_id);
CREATE INDEX ON run USING btree (slurm_job_id);
CREATE INDEX ON run USING btree (experiment_id, record_timestamp);

