select
    kind,
    val
from
(
    select
        pid.parameter_id,
        to_jsonb(kind) kind,
        COALESCE(to_jsonb(bool_value), to_jsonb(integer_value), to_jsonb(real_value), to_jsonb(string_value)) val
    FROM
        (
            select 
                unnest(experiment_parameters) parameter_id
            from 
                experiment_ e
            where
                e.experiment_id = 2255259
         ) pid
         LEFT JOIN parameter_ p ON (p.id = pid.parameter_id)
    ORDER BY
        pid.parameter_id 
) x
;

SELECT 
    COUNT(1)
FROM
    experiment_ e
WHERE
    experiment_parameters && (
        SELECT array_agg(id)
        FROM parameter_ p
        WHERE p.kind = 'batch'
    );
    


SELECT 
    MIN(experiment_id),
    MAX(experiment_id)
FROM
    experiment_ e
WHERE
    (
        SELECT array_agg(id)
        FROM parameter_ p
        WHERE p.kind = 'batch'
    ) @> experiment_parameters


SELECT
    *
FROM 
    experiment_ e
WHERE
    e.experiment_parameters @>
(select
    array_agg(parameter_id)
from
(
    select
        pid.parameter_id
    FROM
        (
            select 
                unnest(experiment_parameters) parameter_id
            from 
                experiment_ e
            where
                e.experiment_id = 2255259
        ) pid,
        parameter_ p
    WHERE
        p.id = pid.parameter_id 
        AND p.kind <> 'batch'
    ORDER BY
        pid.parameter_id 
) x)
;


SELECT
    COUNT(1) num,
    (select
        array_agg(id)
    from
    (
        select
            ep.id
        FROM
            unnest(experiment_parameters) ep(id),
            parameter_ p
        WHERE
            p.id = ep.id
            AND p.kind <> 'batch'
        ORDER BY
            ep.id DESC
    )x ) params,
    array_agg(
        (select 
            string_value
        from
            unnest(experiment_parameters) ep(id),
            parameter_ p
        WHERE
            p.id = ep.id
            AND p.kind = 'batch'
        limit 1) 
    ) batches
FROM 
    experiment_ e
GROUP BY params
ORDER BY num DESC
LIMIT 1000
;


WITH
keys_to_remove AS
(
   SELECT id FROM parameter_ where kind = 'batch'
),
mapping AS 
(
    SELECT
        unnest(targets) src_id, *
    FROM
        (
            SELECT
                MIN(experiment_id) dst_id,
                (select
                    array_agg(id)
                from
                (
                    select
                        ep.id
                    FROM
                        unnest(experiment_parameters) ep(id)
                    WHERE
                        NOT EXISTS (SELECT 1 FROM keys_to_remove WHERE keys_to_remove.id = ep.id)
                    ORDER BY
                        ep.id DESC
                )x ) new_params,
                array_agg(experiment_id) targets
            FROM 
                experiment_ e
            WHERE
                (
                    SELECT array_agg(keys_to_remove.id)
                    FROM keys_to_remove
                ) && e.experiment_parameters
            GROUP BY new_params
        ) x
    LIMIT 1000
)
SELECT * FROM mapping
;


WITH
keys_to_remove AS
(
   SELECT id FROM parameter_ where kind = 'batch'
),
experiment_map AS 
(
    SELECT
        unnest(targets) src_id, dst_id, new_params
    FROM
        (
            SELECT
                MIN(experiment_id) dst_id,
                (select
                    array_agg(id)
                from
                (
                    select
                        ep.id
                    FROM
                        unnest(experiment_parameters) ep(id)
                    WHERE
                        NOT EXISTS (SELECT 1 FROM keys_to_remove WHERE keys_to_remove.id = ep.id)
                    ORDER BY
                        ep.id DESC
                )x ) new_params,
                array_agg(experiment_id) targets
            FROM 
                experiment_ e
            WHERE
                (
                    SELECT array_agg(keys_to_remove.id)
                    FROM keys_to_remove
                ) && e.experiment_parameters
            GROUP BY new_params
        ) x
),
updated AS
(
    SELECT COUNT(1) num FROM experiment_ e, experiment_map
    WHERE
        experiment_map.src_id = experiment_map.dst_id
        AND experiment_map.src_id = e.experiment_id
),
deleted AS
(
    SELECT COUNT(1) num FROM experiment_ e, experiment_map
    WHERE
        experiment_map.src_id <> experiment_map.dst_id
        AND experiment_map.src_id = e.experiment_id
),
updated_run AS
(
    SELECT COUNT(1) num FROM run_ r, experiment_map
    WHERE
        experiment_map.src_id = experiment_map.dst_id
        AND experiment_map.src_id = r.experiment_id
),
updated_experiment_summary AS
(
    SELECT COUNT(1) num FROM experiment_summary_ e, experiment_map
    WHERE
        experiment_map.src_id = experiment_map.dst_id
        AND experiment_map.src_id = e.experiment_id
),
deleted_experiment_summary AS
(
    SELECT COUNT(1) num FROM experiment_summary_ e, experiment_map
    WHERE
        experiment_map.src_id <> experiment_map.dst_id
        AND experiment_map.src_id = e.experiment_id
)
SELECT updated.num updated, deleted.num deleted, updated_run.num updated_run, updated_experiment_summary.num updated_experiment_summary, deleted_experiment_summary.num deleted_experiment_summary FROM 
updated, deleted, updated_run, updated_experiment_summary, deleted_experiment_summary
;


WITH
keys_to_remove AS
(
   SELECT id FROM parameter_ where kind = 'batch'
),
experiment_map AS 
(
    SELECT
        unnest(targets) src_id, dst_id, new_params
    FROM
        (
            SELECT
                MIN(experiment_id) dst_id,
                (select
                    array_agg(id)
                from
                (
                    select
                        ep.id
                    FROM
                        unnest(experiment_parameters) ep(id)
                    WHERE
                        NOT EXISTS (SELECT 1 FROM keys_to_remove WHERE keys_to_remove.id = ep.id)
                    ORDER BY
                        ep.id DESC
                )x ) new_params,
                array_agg(experiment_id) targets
            FROM 
                experiment_ e
            WHERE
                (
                    SELECT array_agg(keys_to_remove.id)
                    FROM keys_to_remove
                ) && e.experiment_parameters
            GROUP BY new_params
        ) x
),
updated AS
(
    UPDATE experiment_ e SET
        experiment_parameters = experiment_map.new_params
    FROM experiment_map
    WHERE
        experiment_map.src_id = experiment_map.dst_id
        AND experiment_map.src_id = e.experiment_id
),
deleted AS
(
    DELETE FROM experiment_ e 
    USING experiment_map
    WHERE
        experiment_map.src_id <> experiment_map.dst_id
        AND experiment_map.src_id = e.experiment_id
),
updated_run AS
(
    UPDATE run_ r SET
        experiment_id = experiment_map.dst_id
    FROM experiment_map
    WHERE
        experiment_map.src_id = experiment_map.dst_id
        AND experiment_map.src_id = r.experiment_id
),
updated_experiment_summary AS
(
    UPDATE experiment_summary_ e SET
        experiment_parameters = experiment_map.new_params
    FROM experiment_map
    WHERE
        experiment_map.src_id = experiment_map.dst_id
        AND experiment_map.src_id = e.experiment_id
),
deleted_experiment_summary AS
(
    DELETE FROM experiment_summary_ e 
    USING experiment_map
    WHERE
        experiment_map.src_id <> experiment_map.dst_id
        AND experiment_map.src_id = e.experiment_id
)
SELECT * FROM experiment_map LIMIT 10
;





