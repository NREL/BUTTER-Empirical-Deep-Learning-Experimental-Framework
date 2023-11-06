select count(1) num, queue, model, status, error
from
(
select *,
	command->'experiment'->'model'->'type' model
from
	run_status s inner join run_data using (id)
) x
group by queue, model, status, error
order by queue, model, status, error;

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
    queue = 200
    AND status = 3
    ;

SELECT *
FROM run_status
WHERE status = 3
GROUP BY error
ORDER BY error;


select * from run_data where command @@ '$.experiment.type == "LTHExperiment"';