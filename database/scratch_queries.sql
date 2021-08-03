SELECT status, COUNT(*) FROM jobqueue WHERE groupname = 'exp00' GROUP BY status;
SELECT * FROM jobqueue WHERE groupname =  'exp00' AND status = 'running' AND (CURRENT_TIMESTAMP - update_time) > '4 hours';
UPDATE jobqueue SET status = NULL WHERE groupname =  'exp00' AND status = 'running' AND (CURRENT_TIMESTAMP - update_time) > '60 minutes';

SELECT COUNT(*) FROM jobqueue WHERE groupname = 'exp00';

select * from jobqueue where status is not NULL LIMIT 1000;

delete from jobqueue;

SELECT version();

select doc from "log" where "groupname" IN ('exp00', 'exp01') AND "timestamp" > '2000-01-01 00:00:00' ORDER BY id DESC LIMIT 100;

SELECT COUNT(*) FROM log;

UPDATE log
    SET jobid = CAST("doc" -> 'environment' ->> 'SLURM_JOB_ID' AS BIGINT)
    WHERE job IS NOT NULL AND groupname IS NOT NULL;

UPDATE jobqueue
    SET jobid = log.jobid
    FROM (SELECT job, jobid, groupname FROM log) as log
    WHERE log.job = jobqueue.uuid AND log.jobid IS NOT NULL;

SELECT status, retry_count, COUNT(*), AVG((config->>'budget')::int) AS budget FROM jobqueue WHERE groupname = 'exp00' GROUP BY status, retry_count;
SELECT status, retry_count,
       COUNT(*), AVG((config->>'budget')::int) AS budget FROM jobqueue WHERE groupname = 'exp02' GROUP BY status, retry_count;

SELECT status, AVG((config->>'budget')::int) AS budget FROM jobqueue WHERE groupname = 'exp00' AND (CURRENT_TIMESTAMP - update_time) < '8 hours' GROUP BY status;

SELECT status, AVG((config->>'budget')::int) AS budget FROM jobqueue WHERE groupname = 'exp00' AND
                                                                           status = 'running' AND
                                                                           (CURRENT_TIMESTAMP - update_time) > '4 hours'
                                                                           GROUP BY status;



SELECT AVG((config->>'budget')::int) AS budget FROM jobqueue WHERE groupname = 'exp00'
                                                                       AND status = 'running'
alter table log alter column doc type jsonb using doc::jsonb;

alter table jobqueue alter column config type jsonb using config::jsonb;

create index log_doc_index on log using gin(doc);

select doc->'config'->'early_stopping'->'baseline' AS "endpoint" from log limit 100;

SELECT DISTINCT "config.dataset" FROM materialized_experiments_0;



-- CREATE TABLE materialized_experiments_0 AS
-- SELECT
--     *,
--     (doc->>'iterations')::bigint AS "iterations",
--     (doc->'loss')::float AS "loss",
--     CAST((doc->>'num_classes')::float AS BIGINT) AS "num_classes",
--     CAST((doc->>'num_features')::float AS BIGINT) AS "num_features",
--     (doc->>'num_inputs')::bigint AS "num_inputs",
--     (doc->>'num_observations')::bigint AS "num_observations",
--     (doc->>'num_outputs')::bigint AS "num_outputs",
--     (doc->>'num_weights')::bigint AS "num_weights",
--     (doc->>'run_name') AS "run_name",
--     (doc->>'task') AS "task",
--     (doc->'test_loss')::float AS "test_loss",
--     (doc->'val_loss')::float AS "val_loss",
--     (doc->'config'->>'activation') AS "config.activation",
--     (doc->'config'->>'budget')::bigint AS "config.budget",
--     (doc->'config'->>'dataset') AS "config.dataset",
--     (doc->'config'->>'depth') AS "config.depth",
--     (doc->'config'->'early_stopping'->'baseline')::text AS "config.early_stopping.baseline",
--     (doc->'config'->'early_stopping'->'min_delta')::float AS "config.early_stopping.min_delta",
--     (doc->'config'->'early_stopping'->'mode')::text AS "config.early_stopping.mode",
--     (doc->'config'->'early_stopping'->>'monitor') AS "config.early_stopping.monitor",
--     (doc->'config'->'early_stopping'->>'patience')::bigint AS "config.early_stopping.patience",
--     (doc->'config'->>'log') AS "config.log",
--     (doc->'config'->>'mode') AS "config.mode",
--     (doc->'config'->>'name') AS "config.name",
--     (doc->'config'->>'num_hidden')::bigint AS "config.num_hidden",
--     (doc->'config'->>'residual_mode') AS "config.residual_mode",
--     (doc->'config'->>'run_name') AS "config.run_name",
--     (doc->'config'->>'test_split')::float AS "config.test_split",
--     (doc->'config'->>'topology') AS "config.topology",
--     (doc->>'topology') AS "topology"
--     FROM log;

CREATE INDEX materialized_experiments_0_id on materialized_experiments_0 (id);
CREATE INDEX materialized_experiments_0_name on materialized_experiments_0 (name);
CREATE INDEX materialized_experiments_0_num_weights ON materialized_experiments_0 ("num_weights");
CREATE INDEX materialized_experiments_0_run_name ON materialized_experiments_0 ("run_name");
CREATE INDEX materialized_experiments_0_task ON materialized_experiments_0 ("task");
CREATE INDEX materialized_experiments_0_timestamp on materialized_experiments_0 (timestamp);
CREATE INDEX materialized_experiments_0_topology ON materialized_experiments_0 ("topology");

-- CREATE INDEX materialized_experiments_0_groupname on materialized_experiments_0 (groupname);
-- CREATE INDEX materialized_experiments_0_iterations ON materialized_experiments_0 ("iterations");
-- CREATE INDEX materialized_experiments_0_loss ON materialized_experiments_0 ("loss");
-- CREATE INDEX materialized_experiments_0_num_classes ON materialized_experiments_0 ("num_classes");
-- CREATE INDEX materialized_experiments_0_num_features ON materialized_experiments_0 ("num_features");
-- CREATE INDEX materialized_experiments_0_num_inputs ON materialized_experiments_0 ("num_inputs");
-- CREATE INDEX materialized_experiments_0_num_observations ON materialized_experiments_0 ("num_observations");
-- CREATE INDEX materialized_experiments_0_num_outputs ON materialized_experiments_0 ("num_outputs");
-- CREATE INDEX materialized_experiments_0_test_loss ON materialized_experiments_0 ("test_loss");
-- CREATE INDEX materialized_experiments_0_val_loss ON materialized_experiments_0 ("val_loss");
-- CREATE INDEX materialized_experiments_0_config_activation ON materialized_experiments_0 ("config.activation");

CREATE INDEX materialized_experiments_0_config_budget ON materialized_experiments_0 ("config.budget");
CREATE INDEX materialized_experiments_0_config_dataset ON materialized_experiments_0 ("config.dataset");
CREATE INDEX materialized_experiments_0_config_depth ON materialized_experiments_0 ("config.depth");
-- CREATE INDEX materialized_experiments_0_config_early_stopping_baseline ON materialized_experiments_0 ("config.early_stopping.baseline");
-- CREATE INDEX materialized_experiments_0_config_early_stopping_min_delta ON materialized_experiments_0 ("config.early_stopping.min_delta");
-- CREATE INDEX materialized_experiments_0_config_early_stopping_mode ON materialized_experiments_0 ("config.early_stopping.mode");
CREATE INDEX materialized_experiments_0_config_early_stopping_monitor ON materialized_experiments_0 ("config.early_stopping.monitor");
CREATE INDEX materialized_experiments_0_config_early_stopping_patience ON materialized_experiments_0 ("config.early_stopping.patience");
CREATE INDEX materialized_experiments_0_config_log ON materialized_experiments_0 ("config.log");
-- CREATE INDEX materialized_experiments_0_config_mode ON materialized_experiments_0 ("config.mode");
CREATE INDEX materialized_experiments_0_config_name ON materialized_experiments_0 ("config.name");
CREATE INDEX materialized_experiments_0_config_num_hidden ON materialized_experiments_0 ("config.num_hidden");
CREATE INDEX materialized_experiments_0_config_residual_mode ON materialized_experiments_0 ("config.residual_mode");
CREATE INDEX materialized_experiments_0_config_run_name ON materialized_experiments_0 ("config.run_name");
-- CREATE INDEX materialized_experiments_0_config_test_split ON materialized_experiments_0 ("config.test_split");
CREATE INDEX materialized_experiments_0_config_topology ON materialized_experiments_0 ("config.topology");


INSERT INTO materialized_experiments_0
SELECT
    id,
    name,
    timestamp,
    doc,
    job,
    groupname,
    CAST("doc" -> 'environment' ->> 'SLURM_JOB_ID' AS BIGINT) AS jobid,
    (doc->>'iterations')::bigint AS "iterations",
    (doc->'loss')::float AS "loss",
    CAST((doc->>'num_classes')::float AS BIGINT) AS "num_classes",
    CAST((doc->>'num_features')::float AS BIGINT) AS "num_features",
    (doc->>'num_inputs')::bigint AS "num_inputs",
    (doc->>'num_observations')::bigint AS "num_observations",
    (doc->>'num_outputs')::bigint AS "num_outputs",
    (doc->>'num_weights')::bigint AS "num_weights",
    (doc->>'run_name') AS "run_name",
    (doc->>'task') AS "task",
    (doc->'test_loss')::float AS "test_loss",
    (doc->'val_loss')::float AS "val_loss",
    (doc->'config'->>'activation') AS "config.activation",
    (doc->'config'->>'budget')::bigint AS "config.budget",
    (doc->'config'->>'dataset') AS "config.dataset",
    (doc->'config'->>'depth') AS "config.depth",
    (doc->'config'->'early_stopping'->'baseline')::text AS "config.early_stopping.baseline",
    (doc->'config'->'early_stopping'->'min_delta')::float AS "config.early_stopping.min_delta",
    (doc->'config'->'early_stopping'->'mode')::text AS "config.early_stopping.mode",
    (doc->'config'->'early_stopping'->>'monitor') AS "config.early_stopping.monitor",
    (doc->'config'->'early_stopping'->>'patience')::bigint AS "config.early_stopping.patience",
    (doc->'config'->>'log') AS "config.log",
    (doc->'config'->>'mode') AS "config.mode",
    (doc->'config'->>'name') AS "config.name",
    (doc->'config'->>'num_hidden')::bigint AS "config.num_hidden",
    (doc->'config'->>'residual_mode') AS "config.residual_mode",
    (doc->'config'->>'run_name') AS "config.run_name",
    (doc->'config'->>'test_split')::float AS "config.test_split",
    (doc->'config'->>'topology') AS "config.topology",
    (doc->>'topology') AS "topology",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'loss') as v) AS "history_loss",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'hinge') as v) AS "history_hinge",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'accuracy') as v) AS "history_accuracy",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_loss') as v) AS "history_val_loss",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_hinge') as v) AS "history_val_hinge",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_accuracy') as v) AS "history_val_accuracy",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'squared_hinge') as v) AS "history_squared_hinge",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'cosine_similarity') as v) AS "history_cosine_similarity",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_squared_hinge') as v) AS "history_val_squared_hinge",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'mean_squared_error') as v) AS "history_mean_squared_error",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'mean_absolute_error') as v) AS "history_mean_absolute_error",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_cosine_similarity') as v) AS "history_val_cosine_similarity",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_mean_squared_error') as v) AS "history_val_mean_squared_error",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'root_mean_squared_error') as v) AS "history_root_mean_squared_error",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_mean_absolute_error') as v) AS "history_val_mean_absolute_error",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'kullback_leibler_divergence') as v) AS "history_kullback_leibler_divergence",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_root_mean_squared_error') as v) AS "history_val_root_mean_squared_error",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'mean_squared_logarithmic_error') as v) AS "history_mean_squared_logarithmic_error",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_kullback_leibler_divergence') as v) AS "history_val_kullback_leibler_divergence",
    (SELECT array_agg(v::float) AS v FROM jsonb_array_elements(doc->'history'->'val_mean_squared_logarithmic_error') as v) AS "history_val_mean_squared_logarithmic_error"
    FROM log
    WHERE
    log.timestamp > (SELECT MAX(timestamp) FROM materialized_experiments_0) AND
          NOT EXISTS (SELECT id from materialized_experiments_0 WHERE id = log.id);

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
    WHERE groupname = 'exp00'
        AND (status = 'running' AND
              update_time < CURRENT_TIMESTAMP - INTERVAL '4 hours');

UPDATE jobqueue
    SET status = null,
        start_time = null,
        host = null,
        retry_count = retry_count + 1
    WHERE groupname = 'exp00'
        AND (status = 'failed' OR
             (status = 'running' AND
              update_time < CURRENT_TIMESTAMP - INTERVAL '4 hours'));

UPDATE jobqueue
    SET status = null,
        start_time = null,
        host = null,
        retry_count = retry_count + 1
    WHERE groupname = 'exp02'
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