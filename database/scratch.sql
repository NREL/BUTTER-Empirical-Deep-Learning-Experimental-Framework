select distinct(status) from jobqueue where groupname='fixed_3k_1';


SELECT status, COUNT(*) FROM jobqueue WHERE groupname = 'exp00' GROUP BY status;
SELECT * FROM jobqueue WHERE groupname =  'exp00' AND status = 'running' AND (CURRENT_TIMESTAMP - update_time) > '4 hours';
UPDATE jobqueue SET status = NULL WHERE groupname =  'exp00' AND status = 'running' AND (CURRENT_TIMESTAMP - update_time) > '60 minutes';

SELECT COUNT(*), log.doc->'config'->'residual_mode' FROM log
WHERE
    groupname = 'fixed_3k_1'
GROUP BY log.doc->'config'->'residual_mode';

SELECT DISTINCT(jobid) from jobqueue limit 10;

select COUNT(log.id) from log AS log
    where groupname IN ('fixed_3k_1', 'fixed_3k_0', 'fixed_01', 'exp00', 'exp01') AND NOT
    (EXISTS (SELECT id FROM materialized_experiments_3_base AS d WHERE d.id = log.id) AND
     EXISTS (SELECT id FROM materialized_experiments_3_val_loss AS d WHERE d.id = log.id) AND
     EXISTS (SELECT id FROM materialized_experiments_3_history AS d WHERE d.id = log.id));

select * from jobqueue where status is not NULL LIMIT 1000;

select count(*) from jobqueue where (end_time - NOW()) < '15 days';


SELECT pg_size_pretty( pg_total_relation_size('log') );
SELECT pg_size_pretty( pg_total_relation_size('materialized_experiments_0') );
SELECT pg_size_pretty( pg_total_relation_size('materialized_experiments_3_base') );
SELECT pg_size_pretty( pg_total_relation_size('materialized_experiments_3_loss') );
SELECT pg_size_pretty( pg_total_relation_size('materialized_experiments_3_history') );

ALTER TABLE materialized_experiments_3_history ADD COLUMN loss real[];
UPDATE materialized_experiments_3_history h
SET loss = (SELECT l.loss from materialized_experiments_3_loss l WHERE l.id = h.id);
alter table materialized_experiments_3_history alter column loss set not null;
alter table materialized_experiments_3_loss drop column loss;
alter table materialized_experiments_3_loss rename to materialized_experiments_3_val_loss;

ALTER TABLE materialized_experiments_3_history ADD COLUMN val_loss real[];
UPDATE materialized_experiments_3_history h
SET val_loss = (SELECT l.val_loss from materialized_experiments_3_val_loss l WHERE l.id = h.id);
alter table materialized_experiments_3_history alter column val_loss set not null;
vacuum FULL materialized_experiments_3_base;
vacuum FULL materialized_experiments_3_history;
vacuum FULL materialized_experiments_3_val_loss;


SELECT version();

SELECT l.*,a.*
  FROM pg_locks l
  JOIN pg_stat_activity a USING (pid)
 WHERE NOT granted;

SELECT l.*,a.*
  FROM pg_locks l
  JOIN pg_stat_activity a USING (pid)
 WHERE granted
   AND (database,relation) IN (SELECT database,relation
                                 FROM pg_locks WHERE NOT granted);

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

SELECT * FROM pg_stat_activity WHERE datname = 'dmpapps'  ORDER BY backend_start;

SELECT * FROM pg_stat_activity;

SELECT pg_terminate_backend(pid) FROM (VALUES (14951),
(15776)
) AS tab(pid);

select * FROM (VALUES ('!'), (' !'), ('! '), (' '), ('  '), ('0'), ('01'), ('1'), ('2'), (' 1'), (' 2'), ('1 '), ('2 '), ('a'), ('b'), ('z'), (' a'), (' b'), (' z')) AS tab(val) ORDER BY val;

select '0 ' > ' 0', '0 ' < ' 0', '0 ' = ' 0', '0 ' <> ' 0';

select doc from "log" where "groupname" IN ('exp00', 'exp01') AND "timestamp" > '2000-01-01 00:00:00' ORDER BY id DESC LIMIT 100;

SELECT COUNT(*) FROM log;

select ' 0' < '0';

UPDATE jobqueue
    SET
        priority = GREATEST(0, FLOOR(LOG(1000, GREATEST(1, (jobqueue.config->>'budget')::int / 1000))))::text || (MOD((jobqueue.config->>'seed')::int, 10))::text || (MOD(((jobqueue.config->>'seed')::int / 10)::int, 10))::text || (MOD(((jobqueue.config->>'seed')::int / 100)::int, 100))::text || (MOD(((jobqueue.config->>'seed')::int / 1000)::int, 1000))::text
    WHERE
          groupname = 'fixed_3k_1' AND status <> 'done';

SELECT * from jobqueue
    WHERE
          groupname = 'fixed_3k_1' AND
          status IS NULL
    ORDER BY priority
    LIMIT 10;


UPDATE log
    SET jobid = CAST("doc" -> 'environment' ->> 'SLURM_JOB_ID' AS BIGINT)
    WHERE job IS NOT NULL AND groupname IS NOT NULL;

UPDATE jobqueue
    SET jobid = log.jobid
    FROM (SELECT job, jobid, groupname FROM log) as log
    WHERE log.job = jobqueue.uuid AND log.jobid IS NOT NULL;

DO $$ BEGIN
    CREATE TYPE job_status AS ENUM ('queued', 'running', 'failed', 'complete');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DROP type job_status;




SELECT COUNT(*), learning_rate, label_noise, budget
FROM
     materialized_experiments_3_base as mat
WHERE
      "group" = (SELECT id from strings where value ='fixed_3k_1')
GROUP BY learning_rate, label_noise, budget;

SELECT groupname, status, MIN(retry_count) as "min_retry", AVG(retry_count) as "avg_retry", MAX(retry_count) as "max_retry",
       COUNT(*), SUM((EXTRACT(EPOCH FROM end_time) - EXTRACT(EPOCH FROM start_time))) / (60*60*24) as "hours" FROM jobqueue WHERE groupname = 'fixed_3k_1' GROUP BY groupname, status;

SELECT groupname, status, MIN(retry_count) as "min_retry", AVG(retry_count) as "avg_retry", MAX(retry_count) as "max_retry",
       COUNT(*), SUM((EXTRACT(EPOCH FROM end_time) - EXTRACT(EPOCH FROM start_time))) / (60*60*24) as "hours", AVG((config->>'budget')::int) AS budget, AVG((config->>'depth')::int) AS depth FROM jobqueue WHERE groupname = 'fixed_3k_1' GROUP BY groupname, status;

SELECT config->>'topology', config->>'residual_mode', config->>'label_noise', COUNT(*) as count,
       MIN((config->>'budget')::int) as min_budget, AVG((config->>'budget')::int) as avg_budget,  stddev((config->>'budget')::int) as sdev_budget, MAX((config->>'budget')::int) as max_budget,
       MIN((config->>'depth')::int) as min_depth, AVG((config->>'depth')::int) as avg_depth,  stddev((config->>'depth')::int) as sdev_depth, MAX((config->>'depth')::int) as max_depth
       from jobqueue where groupname = 'fixed_3k_1' AND status IN ('failed') AND retry_count >= 0 GROUP BY config->>'topology', config->>'residual_mode', config->>'label_noise' ORDER BY config->>'topology', config->>'residual_mode', config->>'label_noise';


SELECT * from jobqueue where groupname = 'fixed_3k_0' AND status IN ('failed', 'running') AND retry_count >= 0 AND config->>'topology'::varchar = 'exponential' ORDER BY config->>'topology', config->>'residual_mode' LIMIT 100;

SELECT
    COUNT(*) AS "count",
    log.doc->'task' AS "task"
--     (log.doc->'config'->'budget')::bigint AS "budget",
--     (log.doc->'config'->'depth')::int AS "depth",
--     log.doc->'config'->'dataset' AS "dataset",
--     (CASE WHEN jsonb_typeof(log.doc->'test_loss') = 'number' THEN (log.doc->'test_loss')::float END) AS "test_loss",
--     (CASE WHEN jsonb_typeof(log.doc->'val_loss') = 'number' THEN (log.doc->'val_loss')::float END) AS "val_loss",
--     log.doc->'config'->'residual_mode' AS "residual_mode",
--     log.doc->'config'->'test_split' AS "test_split",
--     log.doc->'config'->'topology' AS "topology",
--     log.doc->'config'->'label_noise' as "label_noise",
--     log.doc->'config'->'optimizer'->'config'->'learning_rate' AS "learning_rate"
    FROM
         log
    WHERE
        log.groupname = 'fixed_3k_1' AND
        jsonb_typeof(log.doc->'val_loss') <> 'number'
    GROUP BY
         log.doc->'task';
--          (log.doc->'config'->'dataset'),
--          (log.doc->'config'->'residual_mode'),
--          (log.doc->'config'->'topology'),
--          (log.doc->'config'->'label_noise'),
--          log.doc->'config'->'optimizer'->'config'->'learning_rate',
--          log.doc->'config'->'test_split';


select AVG(CURRENT_TIMESTAMP - update_time), COUNT(*)
FROM
    jobqueue
where
    groupname = 'fixed_3k_0' AND
    status = 'running' AND
    (CURRENT_TIMESTAMP - update_time) > INTERVAL '4 hours';

select (CURRENT_TIMESTAMP - update_time)
FROM
    jobqueue
where
    groupname = 'fixed_3k_0' AND
    status = 'running' AND
--     (CURRENT_TIMESTAMP - update_time) > INTERVAL '4 hours'
    config->>'topology'::varchar <> 'exponential'
ORDER BY (CURRENT_TIMESTAMP - update_time) ASC OFFSET (1020*9) LIMIT 100;

UPDATE jobqueue
    SET status = null,
        start_time = null,
        host = null,
        retry_count = retry_count + 1
    WHERE groupname = 'fixed_3k_1' AND
        status = 'running' AND
        retry_count = 0 AND
        (CURRENT_TIMESTAMP - update_time) > INTERVAL '24 hours';

UPDATE jobqueue
    SET status = null,
        start_time = null,
        host = null,
        retry_count = retry_count + 1
    WHERE groupname = 'fixed_3k_0'
        AND (status = 'failed' OR
             (status = 'running' AND
              (CURRENT_TIMESTAMP - update_time) > INTERVAL '8 hours'));

UPDATE jobqueue
    SET status = 'failed'
    WHERE groupname = 'fixed_3k_0'
        AND (status = 'failed' OR
             (status = 'running' AND
              (CURRENT_TIMESTAMP - update_time) > INTERVAL '2 hours')) AND
          config->>'topology'::varchar = 'exponential';

UPDATE jobqueue
    SET status = null,
        start_time = null,
        host = null,
        retry_count = retry_count + 1
    WHERE groupname = 'fixed_3k_0'
        AND (status = 'failed' OR
             (status = 'running' AND
              (CURRENT_TIMESTAMP - update_time) > INTERVAL '4 hours' AND
              (config->>'budget')::int < 524288 ));

select *, (t.queue_count - t.log_count) as diff
FROM
     (select groupname, count(*) as log_count, (SELECT count(*) from jobqueue where log.groupname = jobqueue.groupname) as queue_count from log group by groupname) as t;

SELECT COUNT(*), "config.dataset" from materialized_experiments_0 WHERE groupname = 'fixed_01' GROUP BY "config.dataset";

SELECT COUNT(*) FROM log WHERE groupname = 'fixed_3k_0';


SELECT groupname, status, MIN(retry_count) as "min_retry", MAX(retry_count) as "max_retry",
       COUNT(*), SUM((end_time - start_time)) as "time", AVG((config->>'budget')::int) AS budget, AVG((config->>'depth')::int) AS depth FROM jobqueue WHERE status = 'done' GROUP BY groupname, status;

SELECT ((config->>'budget')::int) AS budget, ((config->>'depth')::int) AS depth FROM jobqueue WHERE groupname = 'exp06' AND status = 'running';
SELECT (end_time - start_time) , ((config->>'budget')::int) AS budget, ((config->>'depth')::int) AS depth FROM jobqueue WHERE groupname = 'exp06' AND status = 'done' ORDER BY (end_time - start_time) DESC;


SELECT status, AVG((config->>'budget')::int) AS budget FROM jobqueue WHERE groupname = 'exp00' AND (CURRENT_TIMESTAMP - update_time) < '8 hours' GROUP BY status;

SELECT status, AVG((config->>'budget')::int) AS budget FROM jobqueue WHERE groupname = 'fixed_01' AND
                                                                           status = 'running' AND
                                                                           (CURRENT_TIMESTAMP - update_time) > '16 hours'
                                                                           GROUP BY status;

alter table materialized_experiments_0 alter column depth type int using depth::int;

SELECT AVG((config->>'budget')::int) AS budget FROM jobqueue WHERE groupname = 'exp00'
                                                                       AND status = 'running'
alter table log alter column doc type jsonb using doc::jsonb;

alter table jobqueue alter column config type jsonb using config::jsonb;

create index log_doc_index on log using gin(doc);

select doc AS "endpoint" from log limit 100;

SELECT DISTINCT "config.dataset" FROM materialized_experiments_0;


create table materialized_experiments_0
(
    id                                         integer,
    name                                       varchar,
    timestamp                                  timestamp,
    job                                        uuid,
    groupname                                  varchar,
    jobid                                      bigint,
    iterations                                 bigint,
    loss                                       double precision,
    num_classes                                bigint,
    num_features                               bigint,
    num_inputs                                 bigint,
    num_observations                           bigint,
    num_outputs                                bigint,
    num_weights                                bigint,
    run_name                                   text,
    task                                       text,
    test_loss                                  double precision,
    val_loss                                   double precision,
    activation                                 text,
    budget                                     bigint,
    dataset                                    text,
    depth                                      integer,
    "early_stopping.baseline"                  text,
    "early_stopping.min_delta"                 double precision,
    "early_stopping.mode"                      text,
    "early_stopping.monitor"                   text,
    "early_stopping.patience"                  bigint,
    log                                        text,
    mode                                       text,
    "config.name"                              text,
    num_hidden                                 bigint,
    residual_mode                              text,
    test_split                                 double precision,
    topology                                   text,
    history_loss                               double precision[],
    history_hinge                              double precision[],
    history_accuracy                           double precision[],
    history_val_loss                           double precision[],
    history_val_hinge                          double precision[],
    history_val_accuracy                       double precision[],
    history_squared_hinge                      double precision[],
    history_cosine_similarity                  double precision[],
    history_val_squared_hinge                  double precision[],
    history_mean_squared_error                 double precision[],
    history_mean_absolute_error                double precision[],
    history_val_cosine_similarity              double precision[],
    history_val_mean_squared_error             double precision[],
    history_root_mean_squared_error            double precision[],
    history_val_mean_absolute_error            double precision[],
    history_kullback_leibler_divergence        double precision[],
    history_val_root_mean_squared_error        double precision[],
    history_mean_squared_logarithmic_error     double precision[],
    history_val_kullback_leibler_divergence    double precision[],
    history_val_mean_squared_logarithmic_error double precision[],
    job_length                                 interval,
    learning_rate                              double precision,
    label_noise                                double precision
);


create index materialized_experiments_0_task
    on materialized_experiments_0 (task);

create index materialized_experiments_0_id
    on materialized_experiments_0 (id);

create index materialized_experiments_0_timestamp
    on materialized_experiments_0 (timestamp);

create index materialized_experiments_0_groupname
    on materialized_experiments_0 (groupname);

create index materialized_experiments_0_topology_residual_mode_depth
    on materialized_experiments_0 (groupname, dataset, topology, residual_mode, depth);

create index materialized_experiments_0_composite
    on materialized_experiments_0 (groupname, dataset, learning_rate, label_noise, topology, residual_mode, depth);

create index materialized_experiments_0_composite_1
    on materialized_experiments_0 (groupname, dataset, topology, residual_mode, learning_rate, label_noise, depth);

create index materialized_experiments_0_composite_2
    on materialized_experiments_0 (groupname, dataset, topology, residual_mode, learning_rate, label_noise);

create index materialized_experiments_0_composite_3
    on materialized_experiments_0 (groupname, dataset, learning_rate, label_noise, topology, residual_mode);


SELECT COUNT(*), "config.dataset", "config.topology", "config.residual_mode", "config.budget", "config.depth" from materialized_experiments_0 WHERE groupname='fixed_01' GROUP BY "config.dataset", "config.topology", "config.residual_mode", "config.budget", "config.depth" ORDER BY "config.dataset", "config.topology", "config.residual_mode", "config.budget", "config.depth";

SELECT COUNT(*), doc->'config'->>'topology', doc->'config'->>'budget'  from log WHERE groupname='exp05' GROUP BY doc->'config'->>'topology', doc->'config'->>'budget';

SELECT * FROM jobqueue where groupname='exp05' AND status <> 'done' LIMIT 100;

SELECT COUNT(*), config->>'topology' FROM jobqueue where groupname='exp05' GROUP BY config->>'topology';

SELECT * from jobqueue WHERE uuid = '24f9e605-9edd-44bd-a491-294b12d85179';

ALTER TABLE materialized_experiments_0 ADD COLUMN job_length INTERVAL;

UPDATE materialized_experiments_0
    SET job_length = jobqueue.end_time - jobqueue.start_time
FROM
     jobqueue
WHERE jobqueue.uuid = materialized_experiments_0.job;


INSERT INTO COUNT(*), doc.loss, config.budget, config.topology, config.residual_mode, config.dataset, config.test_split, config.label_noise
FROM log,
     jsonb_to_record((log.doc->'config')::jsonb) config(budget int, topology text, residual_mode text, depth int, dataset text, test_split text, label_noise text),
     jsonb_to_record((log.doc)::jsonb) doc(loss float)
WHERE
    log.groupname = 'fixed_3k_1'

create table canonical_strings
(
    id SMALLINT not NULL GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    value text UNIQUE
);

create index canonical_strings_id_value
    on strings (id, value);

create index canonical_strings_value_id
    on strings (value, id);

INSERT INTO strings (value)
    SELECT DISTINCT(topology)
    FROM materialized_experiments_0
ON CONFLICT (value) DO NOTHING;

INSERT INTO strings (value)
    SELECT DISTINCT((log.doc->'config'->>'dataset'))
    FROM log
ON CONFLICT (value) DO NOTHING;

SELECT DISTINCT(log.doc->'config'->>'dataset')
FROM materialized_experiments_0;


create table materialized_experiments_1
(
	id int not null PRIMARY KEY,
	job uuid not null,
	timestamp timestamp not null,
	groupname smallint not null,
	dataset smallint not null,
	topology smallint not null,
	residual_mode smallint not null,
	budget bigint not null,
	depth smallint not null,
	joblength interval not null,
	learning_rate real not null,
	label_noise real not null,
    history_loss real[] not null,
    history_hinge real[] not null,
    history_accuracy real[] not null,
    history_val_loss real[] not null,
    history_val_hinge real[] not null,
    history_val_accuracy real[] not null,
    history_squared_hinge real[] not null,
    history_cosine_similarity real[] not null,
    history_val_squared_hinge real[] not null,
    history_mean_squared_error real[] not null,
    history_mean_absolute_error real[] not null,
    history_val_cosine_similarity real[] not null,
    history_val_mean_squared_error real[] not null,
    history_root_mean_squared_error real[] not null,
    history_val_mean_absolute_error real[] not null,
    history_kullback_leibler_divergence real[] not null,
    history_val_root_mean_squared_error real[] not null,
    history_mean_squared_logarithmic_error real[] not null,
    history_val_kullback_leibler_divergence real[] not null,
    history_val_mean_squared_logarithmic_error real[] not null
);


INSERT INTO materialized_experiments_0
SELECT
    log.id AS id,
    log.name AS name,
    log.timestamp AS timestamp,
    log.job AS job,
    log.groupname AS groupname,
    CAST(log."doc" -> 'environment' ->> 'SLURM_JOB_ID' AS BIGINT) AS jobid,
    (log.doc->'iterations')::bigint AS "iterations",
    (CASE WHEN jsonb_typeof(log.doc->'loss') = 'number' THEN (log.doc->'loss')::float END)  AS "loss",
    CAST((log.doc->>'num_classes')::float AS BIGINT) AS "num_classes",
    CAST((log.doc->>'num_features')::float AS BIGINT) AS "num_features",
    (log.doc->'num_inputs')::bigint AS "num_inputs",
    (log.doc->'num_observations')::bigint AS "num_observations",
    (log.doc->'num_outputs')::bigint AS "num_outputs",
    (log.doc->'num_weights')::bigint AS "num_weights",
    (log.doc->>'run_name') AS "run_name",
    (log.doc->>'task') AS "task",
    (CASE WHEN jsonb_typeof(log.doc->'test_loss') = 'number' THEN (log.doc->'test_loss')::float END) AS "test_loss",
    (CASE WHEN jsonb_typeof(log.doc->'val_loss') = 'number' THEN (log.doc->'val_loss')::float END) AS "val_loss",
    (log.doc->'config'->>'activation') AS "activation",
    (log.doc->'config'->'budget')::bigint AS "budget",
    (log.doc->'config'->>'dataset') AS "dataset",
    (log.doc->'config'->'depth')::int AS "depth",
    (log.doc->'config'->'early_stopping'->>'baseline') AS "early_stopping.baseline",
    (CASE WHEN jsonb_typeof(log.doc->'config'->'early_stopping'->'min_delta') = 'number' THEN (log.doc->'config'->'early_stopping'->'min_delta')::float END) AS "early_stopping.min_delta",
    (log.doc->'config'->'early_stopping'->>'mode') AS "early_stopping.mode",
    (log.doc->'config'->'early_stopping'->>'monitor') AS "early_stopping.monitor",
    (log.doc->'config'->'early_stopping'->'patience')::bigint AS "early_stopping.patience",
    (log.doc->'config'->>'log') AS "log",
    (log.doc->'config'->>'mode') AS "mode",
    (log.doc->'config'->>'name') AS "name",
    (log.doc->'config'->'num_hidden')::bigint AS "num_hidden",
    (log.doc->'config'->>'residual_mode') AS "residual_mode",
    (CASE WHEN jsonb_typeof(log.doc->'config'->'test_split') = 'number' THEN (log.doc->'config'->'test_split')::float END) AS "test_split",
    (log.doc->'config'->>'topology') AS "topology",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'loss') as v) AS "history_loss",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'hinge') as v) AS "history_hinge",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'accuracy') as v) AS "history_accuracy",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_loss') as v) AS "history_val_loss",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_hinge') as v) AS "history_val_hinge",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_accuracy') as v) AS "history_val_accuracy",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'squared_hinge') as v) AS "history_squared_hinge",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'cosine_similarity') as v) AS "history_cosine_similarity",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_squared_hinge') as v) AS "history_val_squared_hinge",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'mean_squared_error') as v) AS "history_mean_squared_error",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'mean_absolute_error') as v) AS "history_mean_absolute_error",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_cosine_similarity') as v) AS "history_val_cosine_similarity",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_mean_squared_error') as v) AS "history_val_mean_squared_error",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'root_mean_squared_error') as v) AS "history_root_mean_squared_error",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_mean_absolute_error') as v) AS "history_val_mean_absolute_error",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'kullback_leibler_divergence') as v) AS "history_kullback_leibler_divergence",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_root_mean_squared_error') as v) AS "history_val_root_mean_squared_error",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'mean_squared_logarithmic_error') as v) AS "history_mean_squared_logarithmic_error",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_kullback_leibler_divergence') as v) AS "history_val_kullback_leibler_divergence",
    (SELECT array_agg(CASE WHEN jsonb_typeof(v) = 'number' THEN v::float END) AS v FROM jsonb_array_elements(log.doc->'history'->'val_mean_squared_logarithmic_error') as v) AS "history_val_mean_squared_logarithmic_error",
    (jobqueue.end_time - jobqueue.start_time) AS job_length,
    (CASE WHEN jsonb_typeof(log.doc->'config'->'label_noise') = 'number' THEN (log.doc->'config'->'label_noise')::float END) AS "label_noise",
    (CASE WHEN jsonb_typeof(log.doc->'config'->'optimizer'->'config'->'learning_rate') = 'number' THEN (log.doc->'config'->'optimizer'->'config'->'learning_rate')::float END) AS "learning_rate"
    FROM
         log,
         jobqueue
    WHERE
        jobqueue.uuid = log.job AND
        log.timestamp > (SELECT MAX(timestamp) FROM materialized_experiments_0);
--         AND NOT EXISTS (SELECT id from materialized_experiments_0 WHERE id = log.id);


SELECT
    (log.doc->>'loss')::float AS "loss",
    (log.doc->>'test_loss')::float AS "test_loss",
    (log.doc->>'val_loss')::float AS "val_loss",
    (log.doc->'config'->'early_stopping'->>'min_delta')::float AS "early_stopping.min_delta",
    (log.doc->'config'->>'test_split')::float AS "test_split"
    FROM
         log,
         jobqueue
    WHERE
        jobqueue.uuid = log.job AND
        log.timestamp > (SELECT MAX(timestamp) FROM materialized_experiments_0) AND log.groupname = 'fixed_3k_1';

SELECT
   (jsonb_typeof(log.doc->'loss') = 'null'),
    (CASE WHEN jsonb_typeof(log.doc->'loss') = 'number' THEN (log.doc->'loss')::float END) AS "loss",
    (jsonb_typeof(log.doc->'test_loss') = 'null'),
    (jsonb_typeof(log.doc->'val_loss') = 'null'),
    (jsonb_typeof(log.doc->'config'->'early_stopping'->'min_delta') = 'null'),
    (jsonb_typeof(log.doc->'config'->'test_split') = 'null'),
    (jsonb_typeof(log.doc->'config'->'label_noise') = 'null'),
    (jsonb_typeof(log.doc->'config'->'optimizer'->'config'->'learning_rate') = 'null'),
    log.doc
    FROM
         log,
         jobqueue
    WHERE
        jobqueue.uuid = log.job AND
        log.timestamp > (SELECT MAX(timestamp) FROM materialized_experiments_0) AND log.groupname = 'fixed_3k_1';
--         AND (
--             (jsonb_typeof(log.doc->'loss') = 'null') OR
--             (jsonb_typeof(log.doc->'test_loss') = 'null') OR
--             (jsonb_typeof(log.doc->'val_loss') = 'null') OR
--             (jsonb_typeof(log.doc->'config'->'early_stopping'->'min_delta') = 'null') OR
--             (jsonb_typeof(log.doc->'config'->'test_split') = 'null') OR
--             (jsonb_typeof(log.doc->'config'->'label_noise') = 'null') OR
--             (jsonb_typeof(log.doc->'config'->'optimizer'->'config'->'learning_rate') = 'null')
--         );


(CASE WHEN log.doc->'config'->'label_noise' IS NOT NULL AND jsonb_typeof(log.doc->'config'->'label_noise') = 'number' THEN (log.doc->'config'->>'label_noise')::float END) AS "label_noise",
    (CASE WHEN log.doc->'config'->'optimizer'->'config'->'learning_rate' IS NOT NULL AND jsonb_typeof(log.doc->'config'->'optimizer'->'config'->'learning_rate') = 'number' THEN (log.doc->'config'->'optimizer'->'config'->>'learning_rate')::float END) AS "learning_rate"

vacuum materialized_experiments_0;
vacuum log;
vacuum jobqueue;

update log as mat
    set groupname = (SELECT groupname from jobqueue WHERE jobqueue.uuid = mat.job)
    WHERE mat.groupname IS NULL;

SELECT * FROM log
    WHERE
    log.timestamp > (SELECT MAX(timestamp) FROM materialized_experiments_0);

VACUUM materialized_experiments_0;
--   "loss"
--   "hinge"
--   "accuracy"
--   "val_loss"
--   "val_hinge"
--   "val_accuracy"
--   "squared_hinge"
--   "cosine_similarity"
--   "val_squared_hinge"
--   "mean_squared_error"
--   "mean_absolute_error"
--   "val_cosine_similarity"
--   "val_mean_squared_error"
--   "root_mean_squared_error"
--   "val_mean_absolute_error"
--   "kullback_leibler_divergence"
--   "val_root_mean_squared_error"
--   "mean_squared_logarithmic_error"
--   "val_kullback_leibler_divergence"
--   "val_mean_squared_logarithmic_error"

CREATE TABLE materialized_experiments_0_history (
    id INTEGER,
    type VARCHAR,
    iteration INTEGER,
    value FLOAT,
    PRIMARY KEY (id, type, iteration)
);

INSERT INTO materialized_experiments_0_history
    (SELECT id,
       history.key AS type,
       element.ORDINALITY AS iteration,
       element.value::float as value
    FROM materialized_experiments_0,
     jsonb_each(doc->'history') AS history,
     jsonb_array_elements(history.value) with ORDINALITY AS element);

SELECT id,
       history.key AS type,
       element.ORDINALITY AS iteration,
       element.value::float as value
    FROM materialized_experiments_0,
     jsonb_each(doc->'history') AS history,
     jsonb_array_elements(history.value) with ORDINALITY AS element LIMIT 100000;
-- CREATE TABLE materialized_experiments_0_history AS
--     (SELECT id,
--        history.key AS type,
--        element.ORDINALITY AS iteration,
--        element.value::float as value
--     FROM materialized_experiments_0,
--      jsonb_each(doc->'history') AS history,
--      jsonb_array_elements(history.value) with ORDINALITY AS element);

DROP TABLE materialized_experiments_0_history;
DELETE FROM materialized_experiments_0_history;
VACUUM materialized_experiments_0_history;
ALTER TABLE materialized_experiments_0_history ADD PRIMARY KEY (id, type, iteration);

-- SELECT id,
--        history.key AS type,
--        array_agg(element.value::float) as value
--     FROM materialized_experiments_0,
--      jsonb_each(doc->'history') AS history,
--      jsonb_array_elements(history.value) AS element
--     GROUP BY (id, type)
--     LIMIT 100;

alter table materialized_experiments_0 add history_loss float[];
alter table materialized_experiments_0 add history_hinge float[];
alter table materialized_experiments_0 add history_accuracy float[];
alter table materialized_experiments_0 add history_val_loss float[];
alter table materialized_experiments_0 add history_val_hinge float[];
alter table materialized_experiments_0 add history_val_accuracy float[];
alter table materialized_experiments_0 add history_squared_hinge float[];
alter table materialized_experiments_0 add history_cosine_similarity float[];
alter table materialized_experiments_0 add history_val_squared_hinge float[];
alter table materialized_experiments_0 add history_mean_squared_error float[];
alter table materialized_experiments_0 add history_mean_absolute_error float[];
alter table materialized_experiments_0 add history_val_cosine_similarity float[];
alter table materialized_experiments_0 add history_val_mean_squared_error float[];
alter table materialized_experiments_0 add history_root_mean_squared_error float[];
alter table materialized_experiments_0 add history_val_mean_absolute_error float[];
alter table materialized_experiments_0 add history_kullback_leibler_divergence float[];
alter table materialized_experiments_0 add history_val_root_mean_squared_error float[];
alter table materialized_experiments_0 add history_mean_squared_logarithmic_error float[];
alter table materialized_experiments_0 add history_val_kullback_leibler_divergence float[];
alter table materialized_experiments_0 add history_val_mean_squared_logarithmic_error float[];

UPDATE materialized_experiments_0 SET
    history_loss = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'loss') as v),
    history_hinge = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'hinge') as v),
    history_accuracy = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'accuracy') as v),
    history_val_loss = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_loss') as v),
    history_val_hinge = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_hinge') as v),
    history_val_accuracy = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_accuracy') as v),
    history_squared_hinge = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'squared_hinge') as v),
    history_cosine_similarity = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'cosine_similarity') as v),
    history_val_squared_hinge = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_squared_hinge') as v),
    history_mean_squared_error = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'mean_squared_error') as v),
    history_mean_absolute_error = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'mean_absolute_error') as v),
    history_val_cosine_similarity = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_cosine_similarity') as v),
    history_val_mean_squared_error = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_mean_squared_error') as v),
    history_root_mean_squared_error = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'root_mean_squared_error') as v),
    history_val_mean_absolute_error = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_mean_absolute_error') as v),
    history_kullback_leibler_divergence = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'kullback_leibler_divergence') as v),
    history_val_root_mean_squared_error = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_root_mean_squared_error') as v),
    history_mean_squared_logarithmic_error = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'mean_squared_logarithmic_error') as v),
    history_val_kullback_leibler_divergence = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_kullback_leibler_divergence') as v),
    history_val_mean_squared_logarithmic_error = (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_mean_squared_logarithmic_error') as v);

SELECT "id", "name", "timestamp", "job", "groupname", "jobid",
           "iterations", "loss", "num_classes", "num_features", "num_inputs",
           "num_observations", "num_outputs", "num_weights", "run_name", "task",
           "test_loss", "val_loss", "config.activation", "config.budget",
           "config.dataset", "config.depth", "config.early_stopping.baseline",
           "config.early_stopping.min_delta", "config.early_stopping.mode",
           "config.early_stopping.monitor", "config.early_stopping.patience",
           "config.log", "config.mode", "config.name", "config.num_hidden",
           "config.residual_mode", "config.run_name", "config.test_split",
           "config.topology", "topology" FROM "materialized_experiments_0" WHERE "groupname" IN ('exp00', 'exp01') AND "config.dataset" = '537_houses';


SELECT COUNT(*) FROM materialized_experiments_0 WHERE "config.dataset" = '537_houses' AND "config.topology" = 'rectangle' AND "config.residual_mode" = 'none';

SELECT * from materialized_experiments_0 WHERE "config.dataset" = '537_houses';

UPDATE jobqueue
    SET status = null,
        start_time = null,
        host = null,
        retry_count = retry_count + 1
    WHERE groupname = 'fixed_3k_1'
        AND ((status = 'running' AND
              update_time < CURRENT_TIMESTAMP - INTERVAL '12 hours') OR status = 'failed');


UPDATE jobqueue
    SET status = null,
        start_time = null,
        host = null,
        retry_count = retry_count + 1
    WHERE groupname = 'fixed_3k_1'
        AND (status = 'running');



UPDATE jobqueue
    SET status = null,
        start_time = null,
        host = null,
        retry_count = retry_count + 1,
        priority = CURRENT_TIMESTAMP::varchar
    WHERE groupname = 'fixed_01'
        AND status IN ('failed', 'running');

select COUNT(*) FROM log;


SELECT * from log where job IS NOT NULL LIMIT 10;

select * from "log" where groupname IN ('exp00', 'exp01') LIMIT 10;

UPDATE log
    SET
        groupname = jobqueue.groupname,
        job = jobqueue.uuid
    FROM (SELECT uuid, groupname FROM jobqueue) AS jobqueue
    WHERE jobqueue.uuid = log.job AND log.job IS NOT NULL;

VACUUM jobqueue;

alter table log alter column job type uuid using job::uuid;

SELECT DISTINCT groupname from materialized_experiments_0;

SELECT groupname, COUNT(*) from log GROUP BY groupname;

SELECT * from log where groupname is NULL ORDER BY timestamp DESC LIMIT 100;

select * from jobqueue where groupname = 'exp02' limit 10;


SELECT jsonb_set(config, '{early_stopping,patience}', '300'::jsonb) from jobqueue WHERE groupname = 'exp06' AND status IS NULL LIMIT 10;

UPDATE jobqueue
    SET config = jsonb_set(config, '{early_stopping,min_delta}', '1.0e-12'::jsonb)
    WHERE groupname = 'exp06'
        AND status IS NULL;

UPDATE jobqueue
    SET config = jsonb_set(config, '{early_stopping,patience}', '200'::jsonb)
    WHERE groupname = 'exp06'
        AND status IS NULL;

SELECT MAX(iter), MIN(iter), AVG(iter), stddev(iter) FROM (SELECT (doc->'iterations')::int as iter from log where groupname = 'exp05') AS i;

SELECT DISTINCT("groupname") FROM materialized_experiments_0;

SELECT "groupname", MAX("config.early_stopping.patience"), MIN("config.early_stopping.patience"), MAX(("doc"->'config'->'run_config'->'epochs')::bigint), MIN(("doc"->'config'->'run_config'->'epochs')::bigint) FROM materialized_experiments_0 GROUP BY groupname;


select max(max_iterations) as max_iterations,
       max(p95) as p95,
       max(p90) as p90,
       max(p67) as p67,
       max(p50) as p50,
       max(p33) as p33,
       max(avg_percentile) as avg_percentile,
       max(foo.max_patience) as max_patience,
       min(foo.min_patience) as min_patience,
    foo."config.dataset",
    foo."config.budget"
FROM
    (SELECT
        MAX(iterations) as max_iterations,
        (percentile_disc(0.95) within group (order by iterations ASC)) as p95,
        (percentile_disc(0.9) within group (order by iterations ASC)) as p90,
        (percentile_disc(0.67) within group (order by iterations ASC)) as p67,
        (percentile_disc(0.5) within group (order by iterations ASC)) as p50,
        (percentile_disc(0.33) within group (order by iterations ASC)) as p33,
        AVG(iterations) as avg_percentile,
        "config.dataset",
        "config.budget",
        max("config.early_stopping.patience") as max_patience,
        min("config.early_stopping.patience") as min_patience
    FROM
        materialized_experiments_0 t
    WHERE
        "groupname" IN ('exp00','exp01','exp02','exp05','exp06')
    GROUP BY
        "config.topology",
        "config.residual_mode",
        "config.dataset",
        "config.budget") AS foo
    GROUP BY
        foo."config.dataset",
        foo."config.budget";



SELECT min_groups."config.budget", min_groups."config.depth", count(*), AVG(a.value) AS value, a.epoch as epoch
FROM
    (SELECT "config.budget", "config.depth"
     FROM
        (SELECT "config.budget", "config.depth",
               ROW_NUMBER() OVER(PARTITION BY min_values."config.budget" ORDER BY AVG(min_value) ASC) AS rank
        FROM
            (SELECT "config.budget", "config.depth", t.id AS id, MIN(a.val) AS min_value
            FROM
                materialized_experiments_0 t,
                unnest(history_val_loss) WITH ORDINALITY as a(val, epoch)
            WHERE
                "groupname" IN ('fixed_01') and
                "config.dataset"='201_pol' and
                "config.topology"='rectangle' and
                "config.residual_mode"='none' and
                "config.depth" BETWEEN 2 and 20
            GROUP BY "config.budget", "config.depth", t.id) AS min_values
        GROUP BY "config.budget", "config.depth") AS min_groups
        WHERE rank = 1) as min_groups,
    materialized_experiments_0 as t,
    unnest(t.history_val_loss) WITH ORDINALITY as a(value, epoch)
WHERE
    min_groups."config.budget" = t.budget AND
    min_groups."config.depth" = t.depth AND
    t."groupname" IN ('fixed_01')and
    t.dataset ='201_pol' and
    t.topology='rectangle' and
    t.residual_mode ='none' and
    t.depth BETWEEN 2 and 20
GROUP BY min_groups."config.budget", min_groups."config.depth", a.epoch;
--     materialized_experiments_0 as t
-- WHERE
--     min_groups."config.depth" = t."config.depth" AND
--     min_groups."config.budget" = t."config.budget" AND
--     min_groups.rank = 1;

-- SELECT "config.budget", "config.depth", AVG(min_value) AS min_value, AVG(a.value) AS value, a.epoch as epoch, count(*) AS count,
--        ROW_NUMBER() OVER(PARTITION BY t."config.budget", t."config.depth", a.epoch ORDER BY AVG(min_value) ASC) AS rank
-- FROM
-- min_values,
-- materialized_experiments_0 as t,
-- unnest(t.history_val_loss) WITH ORDINALITY as a(value, epoch)
-- WHERE
--     min_values.id = t.id
-- GROUP BY t."config.budget", t."config.depth", a.epoch)
-- SELECT * FROM
-- run_groups
-- WHERE
--     rank = 1
-- SELECT "config.budget", "config.depth", AVG(min_value) AS min_value, AVG(a.value) AS value, a.epoch as epoch, count(*) AS count,
--        ROW_NUMBER() OVER(PARTITION BY "config.budget", t."config.depth" ORDER BY AVG(min_value) ASC) AS rank
-- FROM
-- min_values,
-- materialized_experiments_0 as t,
-- unnest(t.history_val_loss) WITH ORDINALITY as a(value, epoch)
-- WHERE
--     min_values.id = t.id AND
--     rank = 1
-- GROUP BY t."config.budget", t."config.depth";


WITH summary AS (
    SELECT "config.budget", AVG(a.val) AS value, count(a.val), a.epoch, "config.depth",
        ROW_NUMBER() OVER(PARTITION BY epoch, "config.budget", "config.depth" ORDER BY MIN(a.val) ASC) AS rank
FROM
    materialized_experiments_0 t,
    unnest(history_val_loss) WITH ORDINALITY as a(val, epoch)
WHERE
    "groupname" IN ('fixed_01') and
    "config.dataset"='201_pol' and
    "config.topology"='rectangle' and
    "config.residual_mode"='none' and
    "config.depth" BETWEEN 2 and 20
GROUP BY epoch, "config.budget", "config.depth"
)
SELECT * FROM summary
    WHERE rank = 1;

SELECT COUNT(*) FROM log;

SELECT *, time_per_epoch * 1000 as time FROM (
                               select t.dataset,t.budget,AVG((jobqueue.end_time - jobqueue.start_time) / array_length(t.history_val_loss, 1)) as time_per_epoch,stddev(EXTRACT(epoch FROM (jobqueue.end_time - jobqueue.start_time) / array_length(t.history_val_loss, 1))) as stddev
                               FROM jobqueue,
                                    materialized_experiments_0 as t
                               WHERE jobqueue.uuid = t.job
                                 AND jobqueue.groupname IN ('fixed_3k_0')
                               GROUP BY t.dataset,
                                        t.budget
                           ) AS source
ORDER BY time_per_epoch;

SELECT *, time_per_epoch * 1000 as time FROM (
                               select t.dataset,t.budget,AVG((jobqueue.end_time - jobqueue.start_time) / array_length(t.history_val_loss, 1)) as time_per_epoch,stddev(EXTRACT(epoch FROM (jobqueue.end_time - jobqueue.start_time) / array_length(t.history_val_loss, 1))) as stddev
                               FROM jobqueue,
                                    materialized_experiments_0 as t
                               WHERE jobqueue.uuid = t.job
                                 AND jobqueue.groupname IN ('fixed_3k_0')
                               GROUP BY t.dataset,
                                        t.budget
                           ) AS source
ORDER BY time_per_epoch;


SELECT *, time_per_epoch * 1000 as time FROM (
                               select t.dataset,t.budget,t.depth,AVG((jobqueue.end_time - jobqueue.start_time) / array_length(t.history_val_loss, 1)) as time_per_epoch,stddev(EXTRACT(epoch FROM (jobqueue.end_time - jobqueue.start_time) / array_length(t.history_val_loss, 1))) as stddev
                               FROM jobqueue,
                                    materialized_experiments_0 as t
                               WHERE jobqueue.uuid = t.job
                                 AND jobqueue.groupname IN ('fixed_01')
                               GROUP BY
                                        t.dataset,
                                        t.budget,
                                        t.depth
                           ) AS source
ORDER BY time_per_epoch DESC;

drop table mat_core;

create  TABLE mat_core as
    SELECT
        log.id as id,
        log.job as job,
        log.timestamp as timestamp,
        (jobqueue.end_time - jobqueue.start_time) AS job_length,
        config.dataset,
     config.topology,
     config.residual_mode,
     config.budget,
     config.depth,
     (CASE WHEN jsonb_typeof(config.label_noise) = 'number' THEN config.label_noise::real ELSE 0.0 END) AS label_noise,
        history.*
        FROM
             log,
             jobqueue,
             jsonb_to_record(log.doc) as doc(history jsonb, config jsonb),
             jsonb_to_record(doc.config) as config (dataset text, topology text, residual_mode text, budget bigint, depth smallint, label_noise jsonb),
             jsonb_to_record(doc.history) as history (
                loss real[],
                hinge real[],
                accuracy real[],
                val_loss real[],
                val_hinge real[],
                val_accuracy real[],
                squared_hinge real[],
                cosine_similarity real[],
                val_squared_hinge real[],
                mean_squared_error real[],
                mean_absolute_error real[],
                val_cosine_similarity real[],
                val_mean_squared_error real[],
                root_mean_squared_error real[],
                val_mean_absolute_error real[],
                kullback_leibler_divergence real[],
                val_root_mean_squared_error real[],
                mean_squared_logarithmic_error real[],
                val_kullback_leibler_divergence real[],
                val_mean_squared_logarithmic_error real[])
        WHERE
            jobqueue.uuid = log.job AND
            log.groupname = 'fixed_3k_1';

create  TABLE mat_config as SELECT
        log.id,
        log.job,
        log.timestamp,
        log.job_length,
        config.*,
        history.*
        FROM
             mat_core as log,
             jsonb_to_record(log.doc->'config') as config (dataset text, topology text, residual_mode text, budget bigint, depth smallint, label_noise jsonb),
             jsonb_to_record(log.doc->'config') as config (dataset text, topology text, residual_mode text, budget bigint, depth smallint, label_noise jsonb),
             jsonb_to_record(log.doc->'history') as history (
                loss real[],
                hinge real[],
                accuracy real[],
                val_loss real[],
                val_hinge real[],
                val_accuracy real[],
                squared_hinge real[],
                cosine_similarity real[],
                val_squared_hinge real[],
                mean_squared_error real[],
                mean_absolute_error real[],
                val_cosine_similarity real[],
                val_mean_squared_error real[],
                root_mean_squared_error real[],
                val_mean_absolute_error real[],
                kullback_leibler_divergence real[],
                val_root_mean_squared_error real[],
                mean_squared_logarithmic_error real[],
                val_kullback_leibler_divergence real[],
                val_mean_squared_logarithmic_error real[]);


EXPLAIN SELECT
        log.id,
        log.job,
        log.timestamp,
        log.job_length,
        config.*,
        history.*
        FROM
             mat_core as log,
             jsonb_to_record(log.doc->'config') as config (dataset text, topology text, residual_mode text, budget bigint, depth smallint, label_noise jsonb),
             jsonb_to_record(log.doc->'history') as history (
                loss real[],
                hinge real[],
                accuracy real[],
                val_loss real[],
                val_hinge real[],
                val_accuracy real[],
                squared_hinge real[],
                cosine_similarity real[],
                val_squared_hinge real[],
                mean_squared_error real[],
                mean_absolute_error real[],
                val_cosine_similarity real[],
                val_mean_squared_error real[],
                root_mean_squared_error real[],
                val_mean_absolute_error real[],
                kullback_leibler_divergence real[],
                val_root_mean_squared_error real[],
                mean_squared_logarithmic_error real[],
                val_kullback_leibler_divergence real[],
                val_mean_squared_logarithmic_error real[]);


INSERT INTO materialized_experiments_2
    SELECT
        log.id as id,
        log.job as job,
        log.timestamp as timestamp,
        (SELECT id from strings WHERE strings.value = log.groupname) as groupname,
        (SELECT id from strings WHERE strings.value = config.dataset) as dataset,
        (SELECT id from strings WHERE strings.value = config.topology) as topology,
        (SELECT id from strings WHERE strings.value = config.residual_mode) as residual_mode,
        config.budget as budget,
        config.depth as depth,
        (jobqueue.end_time - jobqueue.start_time) AS job_length,
        (config->'optimizer'->'config'->'learning_rate')::real AS learning_rate,
        (CASE WHEN jsonb_typeof(config.label_noise) = 'number' THEN config.label_noise::real ELSE 0.0 END) AS label_noise,
        history.loss as history_loss,
        history.hinge as history_hinge,
        history.accuracy as history_accuracy,
        history.val_loss as history_val_loss,
        history.val_hinge as history_val_hinge,
        history.val_accuracy as history_val_accuracy,
        history.squared_hinge as history_squared_hinge,
        history.cosine_similarity as history_cosine_similarity,
        history.val_squared_hinge as history_val_squared_hinge,
        history.mean_squared_error as history_mean_squared_error,
        history.mean_absolute_error as history_mean_absolute_error,
        history.val_cosine_similarity as history_val_cosine_similarity,
        history.val_mean_squared_error as history_val_mean_squared_error,
        history.root_mean_squared_error as history_root_mean_squared_error,
        history.val_mean_absolute_error as history_val_mean_absolute_error,
        history.kullback_leibler_divergence as history_kullback_leibler_divergence,
        history.val_root_mean_squared_error as history_val_root_mean_squared_error,
        history.mean_squared_logarithmic_error as history_mean_squared_logarithmic_error,
        history.val_kullback_leibler_divergence as history_val_kullback_leibler_divergence,
        history.val_mean_squared_logarithmic_error as history_val_mean_squared_logarithmic_error
        FROM
             log,
             jobqueue,
             jsonb_to_record(log.doc->'config') as config (dataset text, topology text, residual_mode text, budget bigint, depth smallint, label_noise jsonb),
             jsonb_to_record(log.doc->'history') as history (
                loss real[],
                hinge real[],
                accuracy real[],
                val_loss real[],
                val_hinge real[],
                val_accuracy real[],
                squared_hinge real[],
                cosine_similarity real[],
                val_squared_hinge real[],
                mean_squared_error real[],
                mean_absolute_error real[],
                val_cosine_similarity real[],
                val_mean_squared_error real[],
                root_mean_squared_error real[],
                val_mean_absolute_error real[],
                kullback_leibler_divergence real[],
                val_root_mean_squared_error real[],
                mean_squared_logarithmic_error real[],
                val_kullback_leibler_divergence real[],
                val_mean_squared_logarithmic_error real[])
        WHERE
            jobqueue.uuid = log.job AND
            log.groupname = 'fixed_3k_1';