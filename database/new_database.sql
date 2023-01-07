select * from param;


CREATE TABLE parameter
(
    id serial NOT NULL,
    value_type smallint NOT NULL,
    kind text NOT NULL,
    bool_value boolean,
    integer_value bigint,
    double_value double precision,
    string_value text,
    PRIMARY KEY (id)
);

CREATE INDEX on parameter using btree (kind);
CREATE INDEX on parameter using btree (kind, bool_value) INCLUDE (id) WHERE value_type = 0;
CREATE INDEX on parameter using btree (kind, integer_value) INCLUDE (id) WHERE value_type = 1;
CREATE INDEX on parameter using btree (kind, double_value) INCLUDE (id) WHERE value_type = 2;
CREATE INDEX on parameter using btree (kind, string_value) INCLUDE (id) WHERE value_type = 3;


CREATE TABLE experiment
(
    experiment_id serial NOT NULL,
    experiment_parameters integer[] NOT NULL,
    experiment_data jsonb NOT NULL,
    network_structure text,
    PRIMARY KEY (experiment_parameters)
);

CREATE INDEX ON experiment USING gin (experiment_parameters);


CREATE TABLE run
(
    experiment_id integer NOT NULL,
    record_timestamp timestamp,
    job_id uuid,
    run_id uuid,
    
    slurm_job_id bigint,
    task_version smallint,
    num_gpus smallint,
    num_cpus smallint,
    num_nodes smallint, 
    gpu_memory integer,
    host_name text,
    batch text,
    
    run_data jsonb,
    run_history bytea,
    PRIMARY KEY (run_id)
);

CREATE INDEX ON run USING gin (run_data);
CREATE INDEX ON run USING btree (experiment_id);
CREATE INDEX ON run USING btree (job_id);
CREATE INDEX ON run USING btree (slurm_job_id);
CREATE INDEX ON run USING btree (record_timestamp, experiment_id);
CREATE INDEX ON run USING btree (experiment_id, record_timestamp);

