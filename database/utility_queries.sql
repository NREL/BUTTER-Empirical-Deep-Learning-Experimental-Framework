select queue, status, count(1)
FROM run
WHERE queue > 0
group by queue, status;

select queue, status, count(1), left(error_message, 40) err
FROM run
WHERE queue > 0
group by queue, status, err
order by queue, status, err;

update run set error_message = NULL where queue > 0 and status >= 0;

select queue, status, count(1), error_message
FROM run
WHERE queue = 10 and status = -1
group by queue, status, error_message
order by queue, status, error_message;



SELECT
    t.tablename,
    indexname,
    c.reltuples AS num_rows,
    pg_size_pretty(pg_relation_size(quote_ident(t.tablename)::text)) AS table_size,
    pg_size_pretty(pg_relation_size(quote_ident(indexrelname)::text)) AS index_size,
    CASE WHEN indisunique THEN 'Y'
       ELSE 'N'
    END AS UNIQUE,
    idx_scan AS number_of_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_tables t
LEFT OUTER JOIN pg_class c ON t.tablename=c.relname
LEFT OUTER JOIN
    ( SELECT c.relname AS ctablename, ipg.relname AS indexname, x.indnatts AS number_of_columns, idx_scan, idx_tup_read, idx_tup_fetch, indexrelname, indisunique FROM pg_index x
           JOIN pg_class c ON c.oid = x.indrelid
           JOIN pg_class ipg ON ipg.oid = x.indexrelid
           JOIN pg_stat_all_indexes psai ON x.indexrelid = psai.indexrelid )
    AS foo
    ON t.tablename = foo.ctablename
WHERE t.schemaname='public'
ORDER BY 1,2;

SELECT
   relname  as table_name,
   pg_size_pretty(pg_total_relation_size(relid)) As "Total Size",
   pg_size_pretty(pg_relation_size(relid)) as "Core",
   pg_size_pretty(pg_indexes_size(relid)) as "Index",
   pg_size_pretty(pg_table_size(relid) - pg_relation_size(relid) - pg_relation_size(relid, 'vm') - pg_relation_size(relid, 'fsm')) as "TOAST",
   pg_size_pretty(pg_relation_size(relid, 'vm')) as "Visibility Map",
   pg_size_pretty(pg_relation_size(relid, 'fsm')) as "Free Space Map",
   (pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid)) as "Tuples",
   pg_stat_get_live_tuples(relid) as "Live Tuples",
   pg_stat_get_dead_tuples(relid) as "Dead Tuples",
   pg_size_pretty(pg_total_relation_size(relid) / (1 + pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid))) as "Per Tuple",
   pg_size_pretty(pg_relation_size(relid) / (1 + pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid))) as "Core Per Tuple",
   pg_size_pretty(pg_indexes_size(relid) / (1 + pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid))) as "Index Per Tuple",
   pg_size_pretty((pg_table_size(relid) - pg_relation_size(relid) - pg_relation_size(relid, 'vm') - pg_relation_size(relid, 'fsm')) / (1 + pg_stat_get_live_tuples(relid) + pg_stat_get_dead_tuples(relid))) as "TOAST Per Tuple"
   FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;


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

SELECT *
FROM pg_stat_activity
WHERE usename = 'dmpappsops'
ORDER BY state, query_start desc;

SELECT *
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name   = 'materialized_experiments_2'
     ;

delete from run_status
WHERE
    queue IN (10)
    ;

UPDATE run SET
	status = 0
WHERE status IN (-1, 1) AND queue > 0 AND (NOW() - update_time) > '1 second'::interval;

UPDATE run_status
    SET status = 0
WHERE
    queue IN (10, 11)
    ;


delete from history h
where
	NOT EXISTS (SELECT 1 FROM run_status s WHERE s.id = h.id)
	OR EXISTS (SELECT 1 FROM run_status s WHERE s.id = h.id AND s.status = 0)
	;


select count(1) num, queue, model, status, err
from
(
select
	*,
	left(error, 40) err,
	command->'experiment'->'model'->'type' model
from
	run_status s inner join run_data using (id)
) x
group by queue, model, status, err
order by queue, model, status, err;


select * from run_status left join run_data using (id);

SELECT *
FROM
    run_status
WHERE
    queue = 200
    AND status = 1
    AND (CURRENT_TIMESTAMP - update_time) > '48 hours';

UPDATE run_status
    SET status = 0
WHERE
    queue = 200
    AND status = 1
    AND (CURRENT_TIMESTAMP - update_time) > '48 hours';



UPDATE run_status
    SET status = 0
WHERE
    queue IN (51, 52)
    AND status IN (1, 3)
    AND (CURRENT_TIMESTAMP - update_time) > '1 hours';
    ;

SELECT *
FROM run_status
WHERE status = 3
GROUP BY error
ORDER BY error;


select * from run_data where command @@ '$.experiment.type == "LTHExperiment"';


SELECT
	size,
	depth,
	shape,
	dataset,
	learning_rate,
	batch_size,
	optimizer,
	COUNT(1) num,
	SUM(is_cpu) num_cpu,
	(COUNT(1) - SUM(is_cpu)) num_gpu
FROM
(
	SELECT
		(command->'model'->'size')::int size,
		(command->'model'->'depth')::int depth,
		(command->'model'->'shape')::text shape,
		(command->'dataset'->'name')::text dataset,
		(command->'optimizer'->'learning_rate')::float learning_rate,
		(command->'fit'->'batch_size')::int batch_size,
		(command->'optimizer'->'class')::text optimizer,
		(num_gpus <= 0)::int is_cpu
	FROM
		run,
		job_status,
		job_data
	WHERE TRUE
		AND run.batch like '%energy%'
		AND job_status.id = run.run_id
		AND job_status.id = job_data.id
		AND job_status.status = 2
) x
GROUP BY
	learning_rate,
	batch_size,
	optimizer,
	depth,
	shape,
	dataset,
	size
ORDER BY
	learning_rate,
	batch_size,
	optimizer,
	depth,
	shape,
	dataset,
	size
;


