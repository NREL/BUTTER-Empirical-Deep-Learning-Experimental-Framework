-- This SQL script will initialize a PostgreSQL database to run the experimental framework
-- Executing this script on an empty database will prepare it so that the experimental framework can be run using the database as a job (run) queue, and to store and aggregate run records.
--
-- Tables are:
--  job_status - stores job queueing information, priority, queue, status, error messages, etc
--  job_data - stores static job data, including the information requred to execute a job
--  parameter_ - stores a canonical list of "parameters" (attributes) that experimental runs can have. Parameters are canonicalized in order to have efficient GIN indexing for slicing through run records.
--  experiment_ - stores information about "experiments", which are defined by their unique combination of parameters.
--  run_ - stores individual run records, which record all run information, and also contain an experiment_id foreign key into the experiment_ table.
--  experiment_summary_ - stores per-experiment statistics aggregated over all runs of each experiment.
-- See the readme.md file for more information, and for a link to the dataset and dataset readme which corresponds to this schema.
------------------------------------------------------------------
-- Table: job_status
 -- DROP TABLE IF EXISTS job_status;

CREATE TABLE IF NOT EXISTS job_status ( queue smallint NOT NULL,
                                                       status smallint NOT NULL DEFAULT 0,
                                                                                        priority integer NOT NULL,
                                                                                                         id uuid NOT NULL,
                                                                                                                 start_time timestamp without time zone,
                                                                                                                                                   update_time timestamp without time zone,
                                                                                                                                                                                      worker uuid,
                                                                                                                                                                                      error_count smallint, error text , CONSTRAINT job_status_pkey PRIMARY KEY (id));

-- Index: job_status_priority_idx
 -- DROP INDEX IF EXISTS job_status_priority_idx;

CREATE INDEX IF NOT EXISTS job_status_priority_idx ON job_status USING btree (queue ASC NULLS LAST, priority ASC NULLS LAST)
WHERE status = 0;

-- Index: job_status_update_idx
 -- DROP INDEX IF EXISTS job_status_update_idx;

CREATE INDEX IF NOT EXISTS job_status_update_idx ON job_status USING btree (queue ASC NULLS LAST, status ASC NULLS LAST, update_time ASC NULLS LAST)
WHERE status > 0;

------------------------------------------------------------------
-- Table: job_data
 -- DROP TABLE IF EXISTS job_data;

CREATE TABLE IF NOT EXISTS job_data ( id uuid NOT NULL,
                                              command jsonb NOT NULL,
                                                            parent uuid,
                                                            CONSTRAINT job_data_pkey PRIMARY KEY (id));

-- Index: job_data_command_idx
 -- DROP INDEX IF EXISTS job_data_command_idx;

CREATE INDEX IF NOT EXISTS job_data_command_idx ON job_data USING gin (command) ;

-- Index: job_data_parent_idx
 -- DROP INDEX IF EXISTS job_data_parent_idx;

CREATE INDEX IF NOT EXISTS job_data_parent_idx ON job_data USING btree (parent ASC NULLS LAST, id ASC NULLS LAST)
WHERE parent IS NOT NULL;

------------------------------------------------------------------
-- Table: job_data
 -- DROP TABLE IF EXISTS job_data;

CREATE TABLE IF NOT EXISTS job_data ( id uuid NOT NULL,
                                              command jsonb NOT NULL,
                                                            parent uuid,
                                                            CONSTRAINT job_data_pkey PRIMARY KEY (id));

-- Index: job_data_command_idx
 -- DROP INDEX IF EXISTS job_data_command_idx;

CREATE INDEX IF NOT EXISTS job_data_command_idx ON job_data USING gin (command) ;

-- Index: job_data_expr_expr1_expr2_id_idx
 -- DROP INDEX IF EXISTS job_data_expr_expr1_expr2_id_idx;

CREATE INDEX IF NOT EXISTS job_data_expr_expr1_expr2_id_idx ON job_data USING btree ((command -> 'batch'::text) ASC NULLS LAST, (command -> 'shape'::text) ASC NULLS LAST, (command -> 'dataset'::text) ASC NULLS LAST, id ASC NULLS LAST) ;

-- Index: job_data_id_expr_expr1_expr2_idx
 -- DROP INDEX IF EXISTS job_data_id_expr_expr1_expr2_idx;

CREATE INDEX IF NOT EXISTS job_data_id_expr_expr1_expr2_idx ON job_data USING btree (id ASC NULLS LAST, (command -> 'batch'::text) ASC NULLS LAST, (command -> 'shape'::text) ASC NULLS LAST, (command -> 'dataset'::text) ASC NULLS LAST) ;

-- Index: job_data_parent_idx
 -- DROP INDEX IF EXISTS job_data_parent_idx;

CREATE INDEX IF NOT EXISTS job_data_parent_idx ON job_data USING btree (parent ASC NULLS LAST, id ASC NULLS LAST)
WHERE parent IS NOT NULL;

------------------------------------------------------------------
-- Table: experiment_
 -- DROP TABLE IF EXISTS experiment_;

CREATE TABLE IF NOT EXISTS experiment_ ( experiment_id integer NOT NULL DEFAULT nextval('experiment__experiment_id_seq1'::regclass),
                                                                                experiment_parameters smallint[] NOT NULL,
                                                                                                                 num_free_parameters bigint, network_structure jsonb,
                                                                                                                                             widths integer[], size bigint, relative_size_error real, primary_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                             "300_epoch_sweep" boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                        "30k_epoch_sweep" boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                   learning_rate_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                label_noise_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                           batch_size_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     regularization_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   CONSTRAINT experiment__pkey1 PRIMARY KEY (experiment_parameters));

-- Index: experiment__experiment_id_idx
 -- DROP INDEX IF EXISTS experiment__experiment_id_idx;

CREATE UNIQUE INDEX IF NOT EXISTS experiment__experiment_id_idx ON experiment_ USING btree (experiment_id ASC NULLS LAST) ;

-- Index: experiment__experiment_id_idx1
 -- DROP INDEX IF EXISTS experiment__experiment_id_idx1;

CREATE INDEX IF NOT EXISTS experiment__experiment_id_idx1 ON experiment_ USING hash (experiment_id) ;

-- Index: experiment__experiment_id_idx2
 -- DROP INDEX IF EXISTS experiment__experiment_id_idx2;

CREATE INDEX IF NOT EXISTS experiment__experiment_id_idx2 ON experiment_ USING brin (experiment_id) ;

-- Index: experiment__experiment_parameters_idx
 -- DROP INDEX IF EXISTS experiment__experiment_parameters_idx;

CREATE INDEX IF NOT EXISTS experiment__experiment_parameters_idx ON experiment_ USING gin (experiment_parameters) ;

------------------------------------------------------------------
-- Table: run_
 -- DROP TABLE IF EXISTS run_;

CREATE TABLE IF NOT EXISTS run_ ( experiment_id integer, record_timestamp integer DEFAULT ((date_part('epoch'::text, CURRENT_TIMESTAMP) - (1600000000)::double precision))::integer,
                                                                                          run_id uuid NOT NULL,
                                                                                                      job_id uuid,
                                                                                                      run_parameters smallint[] NOT NULL,
                                                                                                                                test_loss real[], train_loss real[], test_accuracy real[], train_accuracy real[], test_mean_squared_error real[], train_mean_squared_error real[], test_mean_absolute_error real[], train_mean_absolute_error real[], test_root_mean_squared_error real[], train_root_mean_squared_error real[], test_mean_squared_logarithmic_error real[], train_mean_squared_logarithmic_error real[], test_hinge real[], train_hinge real[], test_squared_hinge real[], train_squared_hinge real[], test_cosine_similarity real[], train_cosine_similarity real[], test_kullback_leibler_divergence real[], train_kullback_leibler_divergence real[], platform text , git_hash text , hostname text , slurm_job_id text , seed bigint, save_every_epochs smallint CONSTRAINT run__pkey PRIMARY KEY (run_id));


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_loss
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_loss
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_accuracy
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_accuracy
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_mean_squared_error
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_mean_squared_error
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_mean_absolute_error
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_mean_absolute_error
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_root_mean_squared_error
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_root_mean_squared_error
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_mean_squared_logarithmic_error
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_mean_squared_logarithmic_error
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_hinge
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_hinge
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_squared_hinge
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_squared_hinge
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_cosine_similarity
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_cosine_similarity
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN test_kullback_leibler_divergence
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS run_
ALTER COLUMN train_kullback_leibler_divergence
SET
STORAGE EXTERNAL;

-- Index: run__experiment_id_idx
 -- DROP INDEX IF EXISTS run__experiment_id_idx;

CREATE INDEX IF NOT EXISTS run__experiment_id_idx ON run_ USING hash (experiment_id) ;

-- Index: run__experiment_id_idx1
 -- DROP INDEX IF EXISTS run__experiment_id_idx1;

CREATE INDEX IF NOT EXISTS run__experiment_id_idx1 ON run_ USING btree (experiment_id ASC NULLS LAST) ;

-- Index: run__experiment_id_idx2
 -- DROP INDEX IF EXISTS run__experiment_id_idx2;

CREATE INDEX IF NOT EXISTS run__experiment_id_idx2 ON run_ USING brin (experiment_id) ;

-- Index: run__experiment_id_record_timestamp_idx
 -- DROP INDEX IF EXISTS run__experiment_id_record_timestamp_idx;

CREATE INDEX IF NOT EXISTS run__experiment_id_record_timestamp_idx ON run_ USING btree (experiment_id ASC NULLS LAST, record_timestamp ASC NULLS LAST) ;

-- Index: run__experiment_id_record_timestamp_idx1
 -- DROP INDEX IF EXISTS run__experiment_id_record_timestamp_idx1;

CREATE INDEX IF NOT EXISTS run__experiment_id_record_timestamp_idx1 ON run_ USING brin (experiment_id, record_timestamp) ;

-- Index: run__job_id_idx
 -- DROP INDEX IF EXISTS run__job_id_idx;

CREATE INDEX IF NOT EXISTS run__job_id_idx ON run_ USING btree (job_id ASC NULLS LAST) ;

-- Index: run__record_timestamp_experiment_id_idx
 -- DROP INDEX IF EXISTS run__record_timestamp_experiment_id_idx;

CREATE INDEX IF NOT EXISTS run__record_timestamp_experiment_id_idx ON run_ USING btree (record_timestamp ASC NULLS LAST, experiment_id ASC NULLS LAST) ;

-- Index: run__run_parameters_idx
 -- DROP INDEX IF EXISTS run__run_parameters_idx;

CREATE INDEX IF NOT EXISTS run__run_parameters_idx ON run_ USING gin (run_parameters) ;

------------------------------------------------------------------
-- Table: experiment_summary_
 -- DROP TABLE IF EXISTS experiment_summary_;

CREATE TABLE IF NOT EXISTS experiment_summary_ ( experiment_id integer NOT NULL,
                                                                       update_timestamp integer NOT NULL DEFAULT ((date_part('epoch'::text, CURRENT_TIMESTAMP) - (1600000000)::double precision))::integer,
                                                                                                                 experiment_parameters smallint[] NOT NULL,
                                                                                                                                                  num_runs smallint NOT NULL,
                                                                                                                                                                    num smallint[], test_loss_num_finite smallint[], test_loss_avg real[], test_loss_stddev real[], test_loss_min real[], test_loss_max real[], test_loss_percentile real[], train_loss_num_finite smallint[], train_loss_avg real[], train_loss_stddev real[], train_loss_min real[], train_loss_max real[], train_loss_percentile real[], test_accuracy_avg real[], test_accuracy_stddev real[], train_accuracy_avg real[], train_accuracy_stddev real[], test_mean_squared_error_avg real[], test_mean_squared_error_stddev real[], train_mean_squared_error_avg real[], train_mean_squared_error_stddev real[], test_kullback_leibler_divergence_avg real[], test_kullback_leibler_divergence_stddev real[], train_kullback_leibler_divergence_avg real[], train_kullback_leibler_divergence_stddev real[], num_free_parameters integer, network_structure jsonb,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         widths integer[], size bigint, relative_size_error real, test_loss_median real[], train_loss_median real[], test_accuracy_median real[], train_accuracy_median real[], test_mean_squared_error_median real[], train_mean_squared_error_median real[], train_kullback_leibler_divergence_median real[], test_kullback_leibler_divergence_median real[], external_batch text , primary_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 "300_epoch_sweep" boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "30k_epoch_sweep" boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       learning_rate_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    label_noise_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               batch_size_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         regularization_sweep boolean NOT NULL DEFAULT false,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       test_loss_q1 double precision[], test_loss_q3 double precision[], train_loss_q1 double precision[], train_loss_q3 double precision[], test_accuracy_q1 double precision[], test_accuracy_q3 double precision[], train_accuracy_q1 double precision[], train_accuracy_q3 double precision[], test_mean_squared_error_q1 double precision[], test_mean_squared_error_q3 double precision[], train_mean_squared_error_q1 double precision[], train_mean_squared_error_q3 double precision[], test_mean_absolute_error_q1 double precision[], test_mean_absolute_error_q3 double precision[], train_mean_absolute_error_q1 double precision[], train_mean_absolute_error_q3 double precision[], test_root_mean_squared_error_q1 double precision[], test_root_mean_squared_error_q3 double precision[], train_root_mean_squared_error_q1 double precision[], train_root_mean_squared_error_q3 double precision[], test_mean_squared_logarithmic_error_q1 double precision[], test_mean_squared_logarithmic_error_q3 double precision[], train_mean_squared_logarithmic_error_q1 double precision[], train_mean_squared_logarithmic_error_q3 double precision[], test_hinge_q1 double precision[], test_hinge_q3 double precision[], train_hinge_q1 double precision[], train_hinge_q3 double precision[], test_squared_hinge_q1 double precision[], test_squared_hinge_q3 double precision[], train_squared_hinge_q1 double precision[], train_squared_hinge_q3 double precision[], test_cosine_similarity_q1 double precision[], test_cosine_similarity_q3 double precision[], train_cosine_similarity_q1 double precision[], train_cosine_similarity_q3 double precision[], test_kullback_leibler_divergence_q1 double precision[], test_kullback_leibler_divergence_q3 double precision[], train_kullback_leibler_divergence_q1 double precision[], train_kullback_leibler_divergence_q3 double precision[], CONSTRAINT experiment_summary__pkey1 PRIMARY KEY (experiment_id));


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_loss_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_loss_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_loss_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_loss_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_accuracy_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_accuracy_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_accuracy_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_accuracy_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_mean_squared_error_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_mean_squared_error_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_mean_squared_error_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_mean_squared_error_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_mean_absolute_error_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_mean_absolute_error_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_mean_absolute_error_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_mean_absolute_error_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_root_mean_squared_error_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_root_mean_squared_error_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_root_mean_squared_error_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_root_mean_squared_error_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_mean_squared_logarithmic_error_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_mean_squared_logarithmic_error_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_mean_squared_logarithmic_error_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_mean_squared_logarithmic_error_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_hinge_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_hinge_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_hinge_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_hinge_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_squared_hinge_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_squared_hinge_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_squared_hinge_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_squared_hinge_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_cosine_similarity_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_cosine_similarity_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_cosine_similarity_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_cosine_similarity_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_kullback_leibler_divergence_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN test_kullback_leibler_divergence_q3
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_kullback_leibler_divergence_q1
SET
STORAGE EXTERNAL;


ALTER TABLE IF EXISTS experiment_summary_
ALTER COLUMN train_kullback_leibler_divergence_q3
SET
STORAGE EXTERNAL;

-- Index: experiment_summary__experiment_id_idx1
 -- DROP INDEX IF EXISTS experiment_summary__experiment_id_idx1;

CREATE INDEX IF NOT EXISTS experiment_summary__experiment_id_idx1 ON experiment_summary_ USING hash (experiment_id) ;

-- Index: experiment_summary__experiment_id_update_timestamp_idx
 -- DROP INDEX IF EXISTS experiment_summary__experiment_id_update_timestamp_idx;

CREATE INDEX IF NOT EXISTS experiment_summary__experiment_id_update_timestamp_idx ON experiment_summary_ USING btree (experiment_id ASC NULLS LAST, update_timestamp ASC NULLS LAST) ;

-- Index: experiment_summary__experiment_parameters_idx
 -- DROP INDEX IF EXISTS experiment_summary__experiment_parameters_idx;

CREATE INDEX IF NOT EXISTS experiment_summary__experiment_parameters_idx ON experiment_summary_ USING gin (experiment_parameters) ;

-- Index: experiment_summary__update_timestamp_experiment_id_idx
 -- DROP INDEX IF EXISTS experiment_summary__update_timestamp_experiment_id_idx;

CREATE INDEX IF NOT EXISTS experiment_summary__update_timestamp_experiment_id_idx ON experiment_summary_ USING btree (update_timestamp ASC NULLS LAST, experiment_id ASC NULLS LAST) ;

------------------------------------------------------------------
