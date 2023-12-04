CREATE TABLE public.run
(
    queue smallint NOT NULL DEFAULT 0,
    status smallint NOT NULL DEFAULT 0,
	priority integer NOT NULL DEFAULT 0,
	id uuid NOT NULL,
    start_time timestamp with time zone,
    update_time timestamp with time zone,
    worker_id uuid,
	parent_id uuid,
    experiment_id uuid,
    command jsonb NOT NULL,
	history bytea,
    extended_history bytea,
	error_message text,
    CONSTRAINT run_pkey PRIMARY KEY (id)
)



ALTER TABLE IF EXISTS run
    ALTER COLUMN command SET STORAGE EXTENDED;

ALTER TABLE IF EXISTS run
    ALTER COLUMN error SET STORAGE EXTENDED;

ALTER TABLE IF EXISTS run
    ALTER COLUMN history SET STORAGE EXTERNAL;

ALTER TABLE IF EXISTS run
    ALTER COLUMN extended_history SET STORAGE EXTERNAL;

ALTER TABLE run SET (TOAST_TUPLE_TARGET = 128);

CREATE INDEX run_experiment_id_idx ON run (experiment_id) WHERE experiment_id IS NOT NULL;

CREATE INDEX run_unsummarized_idx ON run (experiment_id)
    WHERE experiment_id IS NOT NULL AND status = 2;

CREATE INDEX run_queue_idx ON run (queue, priority) WHERE status = 0;

CREATE INDEX IF NOT EXISTS run_data_command_ops_idx
    ON public.run_data USING gin
    (command jsonb_path_ops)
    TABLESPACE pg_default;


INSERT INTO run
select
	s.queue,
	(CASE
	 	WHEN s.status >= 3 THEN -s.status
	 	ELSE s.status
	 END
	 ) status,
	s.priority,
	s.id,
	s.start_time,
	s.update_time,
	s.worker worker_id,
	s.parent parent_id,
	s.experiment_id,
	d.command,
	h.history,
	h.extended_history,
	s.error
from
	run_status s inner join run_data d on (s.id=d.id)
	inner join history h on (h.id = s.id)
;