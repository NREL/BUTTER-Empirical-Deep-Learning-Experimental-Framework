delete from run_status
WHERE
    queue IN (10)
    ;

delete from run_data d
WHERE
	NOT EXISTS (select 1 from run_status s where s.id = d.id);

delete from history h
where
	NOT EXISTS (SELECT 1 FROM run_status s where s.status = 2 and s.id = h.id)
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


