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

select e.experiment_id, pc.*
from experiment_ e,
    lateral (
        select
            p.kind, 
            COALESCE(to_jsonb(bool_value), to_jsonb(integer_value), to_jsonb(real_value), to_jsonb(string_value)) val,
            count(1) num_exps,
            array_agg(x.experiment_id) matching_exp_ids,
            array_agg(x.primary_sweep) primary_sweeps
        from
            unnest(e.experiment_parameters) parameter_id,
            parameter_ p,
            experiment_ x
        where
            p.id = parameter_id AND
            x.experiment_parameters @> (
                SELECT array_agg(pid) 
                from (select pid
                    from unnest(e.experiment_parameters) pid
                    where pid <> parameter_id
                    order by pid asc
                      ) x ) AND
            x.experiment_id <> e.experiment_id AND
            x.primary_sweep
        group by p.kind, val
    ) pc
where
    e.experiment_id = 2255259;
    

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








--- CLEAN PARAMETERS:
UPDATE experiment_ e
    SET experiment_parameters = (
            SELECT array_agg(id) 
            FROM (
                SELECT DISTINCT id FROM unnest(experiment_parameters) i(id) 
                WHERE id IS NOT NULL
                ORDER BY id ASC
                ) x
        );


----- DELETE PARAMETERS:




WITH
key_to_remove AS
(
   SELECT DISTINCT id FROM parameter_ where kind IN ('output_activation', 'batch')
   ORDER BY id ASC
),
experiment_map AS
(
    SELECT
        unnest(src_ids) src_id,
        dst_id,
        dst_params
    FROM
        (
            SELECT 
                *,
                (SELECT MIN(id) dst_id FROM unnest(src_ids) id)
            FROM
            (
                SELECT
                    (src_ids || (
                        SELECT array_agg(base.experiment_id)
                        FROM experiment_ base
                        WHERE
                            base.experiment_parameters = dst_params
                    )) src_ids,
                    dst_params
                FROM
                    (
                        SELECT
                            array_agg(src.experiment_id) src_ids,
                            (
                                SELECT array_agg(id) 
                                FROM (
                                    SELECT DISTINCT id FROM unnest(experiment_parameters) i(id) 
                                    WHERE id IS NOT NULL AND NOT EXISTS (SELECT 0 FROM key_to_remove r where r.id = i.id)
                                    ORDER BY id ASC
                                    ) x
                            ) dst_params
                        FROM experiment_ src
                        WHERE experiment_parameters && (SELECT array_agg(id) keys_to_remove from key_to_remove)
                        GROUP BY dst_params
                    ) np
            ) np
       ) np
),
to_update AS
(
    SELECT src_id, dst_params FROM experiment_map WHERE src_id = dst_id
),
to_delete AS
(
    SELECT src_id, dst_id FROM experiment_map WHERE src_id <> dst_id
)
SELECT
    (select count(1) from experiment_map) experiment_map_count,
    (select count(1) from to_update) num_to_update, 
    (select count(1) from to_delete) num_to_delete,
    (select count(1) from experiment_map where dst_id is null)
,* FROM experiment_map LIMIT 1000;

WITH
key_to_remove AS
(
   SELECT DISTINCT id FROM parameter_ where kind IN ('output_activation', 'batch')
   ORDER BY id ASC
),
experiment_map AS
(
    SELECT
        unnest(src_ids) src_id,
        dst_id,
        dst_params
    FROM
        (
            SELECT 
                *,
                (SELECT MIN(id) dst_id FROM unnest(src_ids) id)
            FROM
            (
                SELECT
                    (src_ids || (
                        SELECT array_agg(base.experiment_id)
                        FROM experiment_ base
                        WHERE
                            base.experiment_parameters = dst_params
                    )) src_ids,
                    dst_params
                FROM
                    (
                        SELECT
                            array_agg(src.experiment_id) src_ids,
                            (
                                SELECT array_agg(id) 
                                FROM (
                                    SELECT DISTINCT id FROM unnest(experiment_parameters) i(id) 
                                    WHERE id IS NOT NULL AND NOT EXISTS (SELECT 0 FROM key_to_remove r where r.id = i.id)
                                    ORDER BY id ASC
                                    ) x
                            ) dst_params
                        FROM experiment_ src
                        WHERE experiment_parameters && (SELECT array_agg(id) keys_to_remove from key_to_remove)
                        GROUP BY dst_params
                    ) np
            ) np
       ) np
),
to_update AS
(
    SELECT src_id, dst_params FROM experiment_map WHERE src_id = dst_id
),
to_delete AS
(
    SELECT src_id, dst_id FROM experiment_map WHERE src_id <> dst_id
),
deleted AS
(
    DELETE FROM experiment_ e 
    USING to_delete
    WHERE src_id = experiment_id
),
deleted_experiment_summary AS
(
    DELETE FROM experiment_summary_ e 
    USING to_delete
    WHERE src_id = experiment_id
),
updated AS
(
    UPDATE experiment_ e SET
        experiment_parameters = to_update.dst_params
    FROM to_update
    WHERE src_id = experiment_id
),
updated_experiment_summary AS
(
    UPDATE experiment_summary_ e SET
        experiment_parameters = to_update.dst_params
    FROM to_update
    WHERE src_id = experiment_id
),
updated_run AS
(
    UPDATE run_ r SET
        experiment_id = to_delete.dst_id
    FROM to_delete
    WHERE src_id = experiment_id
)
SELECT
    (select count(1) from experiment_map) experiment_map_count,
    (select count(1) from to_update) num_to_update, 
    (select count(1) from to_delete) num_to_delete
;


----------------


WITH
key_to_remove AS
(
   SELECT id FROM parameter_ where kind = 'output_activation'
)
SELECT
    src_id,
    dst_id,
    new_params
FROM
    (
        SELECT
            (SELECT array_agg(f.id) new_params
              FROM unnest(src.experiment_parameters) f(id)
              WHERE f.id NOT IN (select id from key_to_remove)),
            array_agg(src.experiment_id) src_ids,
            keys_to_remove
        FROM
            (SELECT array_agg(key_to_remove.id) keys_to_remove FROM key_to_remove) x
            INNER JOIN experiment_ src ON (src.experiment_parameters && x.keys_to_remove)
        GROUP BY new_params, keys_to_remove
    ) src,
    LATERAL (
        SELECT MIN(dst.experiment_id) dst_id 
        FROM experiment_ dst 
        WHERE  
            dst.experiment_parameters @> src.new_params
            AND dst.experiment_parameters <@ (src.keys_to_remove || src.new_params)
    ) dst,
    LATERAL unnest(src.src_ids) src_id
;



WITH
key_to_remove AS
(
   SELECT id FROM parameter_ where kind = 'output_activation'
),
experiment_map AS
(
    SELECT
        src_id,
        dst_id,
        new_params
    FROM
        (
        SELECT
            keys_to_remove,
            (SELECT array_agg(f.id) new_params
              FROM unnest(src.experiment_parameters) f(id)
              WHERE f.id NOT IN (select id from key_to_remove)),
            array_agg(src.experiment_id) src_ids
        FROM
            (SELECT array_agg(key_to_remove.id) keys_to_remove FROM key_to_remove) x
            INNER JOIN experiment_ src ON (src.experiment_parameters && x.keys_to_remove)
        GROUP BY keys_to_remove, new_params
    ) src,
    LATERAL (
        SELECT MIN(dst.experiment_id) dst_id 
        FROM experiment_ dst 
        WHERE  
            dst.experiment_parameters @> src.new_params
            AND dst.experiment_parameters <@ (src.keys_to_remove || src.new_params)
    ) dst,
    LATERAL unnest(src.src_ids) src_id
),
to_update AS
(
    SELECT * FROM experiment_map WHERE src_id = dst_id
),
to_delete AS
(
    SELECT * FROM experiment_map WHERE src_id <> dst_id
),
updated_experiment AS
(
    SELECT COUNT(1) num FROM experiment_ e, to_update
    WHERE to_update.src_id = e.experiment_id
),
updated_experiment_summary AS
(
    SELECT COUNT(1) num FROM experiment_summary_ e, to_update
    WHERE to_update.src_id = e.experiment_id
),
deleted_experiment AS
(
    SELECT COUNT(1) num FROM experiment_ e, to_delete
    WHERE to_delete.src_id = e.experiment_id
),
deleted_experiment_summary AS
(
    SELECT COUNT(1) num FROM experiment_summary_ e, to_delete
    WHERE to_delete.src_id = e.experiment_id
),
updated_run AS
(
    SELECT COUNT(1) num FROM run_ r, to_delete
    WHERE r.experiment_id = to_delete.src_id
)
SELECT
    (select count(1) from key_to_remove) num_keys, 
    (select count(1) from experiment_map) experiment_map_count,
    (select count(1) from to_update) num_to_update, 
    (select count(1) from to_delete) num_to_delete, 
    updated_experiment.num updated_experiments, 
    updated_experiment_summary.num updated_summaries,
    deleted_experiment.num deleted_experiments, 
    deleted_experiment_summary.num deleted_summaries,
    updated_run.num updated_runs
FROM 
updated_experiment, updated_experiment_summary, deleted_experiment, deleted_experiment_summary, updated_run 
;


-----------------
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








---