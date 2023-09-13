
CREATE TABLE IF NOT EXISTS history
(
	run_id uuid NOT NULL PRIMARY KEY,
    experiment_id uuid NOT NULL,
    history bytea,
    extended_history bytea
);

ALTER TABLE history
    ALTER COLUMN history SET STORAGE EXTERNAL;

ALTER TABLE history
    ALTER COLUMN extended_history SET STORAGE EXTERNAL;


CREATE INDEX on history using btree (experiment_id);

CREATE TABLE experiment2
(
    experiment_id uuid NOT NULL PRIMARY KEY,
    experiment jsonb NOT NULL,
    most_recent_run timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    num_runs integer NOT NULL,
    old_experiment_id integer,
    by_epoch bytea,
    by_loss bytea,
    by_progress bytea,
    epoch_subset bytea
);

ALTER TABLE experiment2
    ALTER COLUMN experiment SET STORAGE PLAIN,
    ALTER COLUMN by_epoch SET STORAGE EXTERNAL,
    ALTER COLUMN by_loss SET STORAGE EXTERNAL,
    ALTER COLUMN by_progress SET STORAGE EXTERNAL,
    ALTER COLUMN epoch_subset SET STORAGE EXTERNAL
;

CREATE INDEX ON experiment2 USING GIN(experiment jsonb_path_ops);

CREATE TABLE checkpoint
(
    id uuid NOT NULL,
    model_number integer NOT NULL,
    model_epoch integer NOT NULL,
    epoch integer NOT NULL,
    CONSTRAINT checkpoint_pkey PRIMARY KEY (id, model_number, model_epoch)
);

