
CREATE TABLE IF NOT EXISTS experiment_
(
    experiment_id serial NOT NULL,
    experiment_parameters smallint[] NOT NULL PRIMARY KEY,
    num_free_parameters bigint,
    widths smallint[],
    network_structure jsonb
);

alter table experiment_ alter column network_structure set storage EXTENDED;
alter table experiment_ alter column widths set storage EXTENDED;

create unique index on experiment_ (experiment_id);
create index on experiment_ using gin (experiment_parameters);


CREATE TABLE IF NOT EXISTS run_
(
    experiment_id integer,
    record_timestamp integer DEFAULT ((date_part('epoch'::text, CURRENT_TIMESTAMP) - (1600000000)::double precision))::integer,
    run_id uuid NOT NULL PRIMARY KEY,
    job_id uuid,
    run_parameters smallint[] NOT NULL,
    val_loss real[],
    loss real[],
    val_accuracy real[],
    accuracy real[],
    val_mean_squared_error real[],
    mean_squared_error real[],
    val_mean_absolute_error real[],
    mean_absolute_error real[],
    val_root_mean_squared_error real[],
    root_mean_squared_error real[],
    val_mean_squared_logarithmic_error real[],
    mean_squared_logarithmic_error real[],
    val_hinge real[],
    hinge real[],
    val_squared_hinge real[],
    squared_hinge real[],
    val_cosine_similarity real[],
    cosine_similarity real[],
    val_kullback_leibler_divergence real[],
    kullback_leibler_divergence real[],
    platform text,
    git_hash text,
    hostname text,
    slurm_job_id text,
    seed bigint,
    save_every_epochs smallint
);

alter table run_ alter column val_loss set storage EXTERNAL;
alter table run_ alter column loss set storage EXTERNAL;
alter table run_ alter column val_accuracy set storage EXTERNAL;
alter table run_ alter column accuracy set storage EXTERNAL;
alter table run_ alter column val_mean_squared_error set storage EXTERNAL;
alter table run_ alter column mean_squared_error set storage EXTERNAL;
alter table run_ alter column val_mean_absolute_error set storage EXTERNAL;
alter table run_ alter column mean_absolute_error set storage EXTERNAL;
alter table run_ alter column val_root_mean_squared_error set storage EXTERNAL;
alter table run_ alter column root_mean_squared_error set storage EXTERNAL;
alter table run_ alter column val_mean_squared_logarithmic_error set storage EXTERNAL;
alter table run_ alter column mean_squared_logarithmic_error set storage EXTERNAL;
alter table run_ alter column val_hinge set storage EXTERNAL;
alter table run_ alter column hinge set storage EXTERNAL;
alter table run_ alter column val_squared_hinge set storage EXTERNAL;
alter table run_ alter column squared_hinge set storage EXTERNAL;
alter table run_ alter column val_cosine_similarity set storage EXTERNAL;
alter table run_ alter column cosine_similarity set storage EXTERNAL;
alter table run_ alter column val_kullback_leibler_divergence set storage EXTERNAL;
alter table run_ alter column kullback_leibler_divergence set storage EXTERNAL;


-- create index on run_ using hash(job_id);
create index on run_ using hash (experiment_id);
create index on run_ (record_timestamp, experiment_id);
create index on run_ using gin (run_parameters);



CREATE TABLE IF NOT EXISTS parameter_
(
    id smallserial NOT NULL primary key,
    bool_value boolean,
    real_value real,
    integer_value bigint,
    string_value text COLLATE pg_catalog."default",
    kind text COLLATE pg_catalog."default" NOT NULL
);

CREATE UNIQUE INDEX on parameter_ (kind, bool_value) include (id) where bool_value is not null;
CREATE UNIQUE INDEX on parameter_ (kind, real_value) include (id) where real_value is not null;
CREATE UNIQUE INDEX on parameter_ (kind, integer_value) include (id) where integer_value is not null;
CREATE UNIQUE INDEX on parameter_ (kind, string_value) include (id) where string_value is not null;
CREATE UNIQUE INDEX on parameter_ (kind) include (id) where bool_value is null and real_value is null and integer_value is null and string_value is null;



CREATE TABLE IF NOT EXISTS experiment_summary_
(
    experiment_id integer NOT NULL primary key,
    update_timestamp integer NOT NULL DEFAULT ((date_part('epoch'::text, CURRENT_TIMESTAMP) - (1600000000)::double precision))::integer,
    experiment_parameters smallint[] NOT NULL,
    num_runs smallint NOT NULL,
    num_free_parameters integer,
    num smallint[],
    val_loss_num_finite smallint[],
    val_loss_avg real[],
    val_loss_stddev real[],
    val_loss_min real[],
    val_loss_max real[],
    val_loss_percentile real[],
    loss_num_finite smallint[],
    loss_avg real[],
    loss_stddev real[],
    loss_min real[],
    loss_max real[],
    loss_percentile real[],
    val_accuracy_avg real[],
    val_accuracy_stddev real[],
    accuracy_avg real[],
    accuracy_stddev real[],
    val_mean_squared_error_avg real[],
    val_mean_squared_error_stddev real[],
    mean_squared_error_avg real[],
    mean_squared_error_stddev real[],
    val_kullback_leibler_divergence_avg real[],
    val_kullback_leibler_divergence_stddev real[],
    kullback_leibler_divergence_avg real[],
    kullback_leibler_divergence_stddev real[],
    network_structure jsonb,
    widths integer[]
);


CREATE INDEX ON experiment_summary_ USING hash (experiment_id);
CREATE INDEX ON experiment_summary_ USING gin (experiment_parameters);
create index on experiment_summary_ (update_timestamp, experiment_id);




insert into experiment_summary_ (
    experiment_id,
    experiment_parameters,
    num_runs,
    num_free_parameters,
    num,
    val_loss_num_finite,
    val_loss_avg,
    val_loss_stddev,
    val_loss_min,
    val_loss_max,
    val_loss_percentile,
    loss_num_finite,
    loss_avg,
    loss_stddev,
    loss_min,
    loss_max,
    loss_percentile,
    val_accuracy_avg,
    val_accuracy_stddev,
    accuracy_avg,
    accuracy_stddev,
    val_mean_squared_error_avg,
    val_mean_squared_error_stddev,
    mean_squared_error_avg,
    mean_squared_error_stddev,
    val_kullback_leibler_divergence_avg,
    val_kullback_leibler_divergence_stddev,
    kullback_leibler_divergence_avg,
    kullback_leibler_divergence_stddev,
    network_structure,
    widths
)
select
    e.experiment_id,
    experiment_parameters,
    num_runs,
    num_free_parameters,
    num,
    val_loss_num_finite,
    val_loss_avg,
    val_loss_stddev,
    val_loss_min,
    val_loss_max,
    val_loss_percentile,
    
    loss_num_finite,
    loss_avg,
    loss_stddev,
    loss_min,
    loss_max,
    loss_percentile,
    
    val_accuracy_.v_avg val_accuracy_avg, 
    val_accuracy_.v_stddev val_accuracy_stddev, 
    accuracy_.v_avg accuracy_avg, 
    accuracy_.v_stddev accuracy_stddev, 
    val_mean_squared_error_.v_avg val_mean_squared_error_avg, 
    val_mean_squared_error_.v_stddev val_mean_squared_error_stddev, 
    mean_squared_error_.v_avg mean_squared_error_avg, 
    mean_squared_error_.v_stddev mean_squared_error_stddev, 
    val_kullback_leibler_divergence_.v_avg val_kullback_leibler_divergence_avg,
    val_kullback_leibler_divergence_.v_stddev val_kullback_leibler_divergence_stddev, 
    kullback_leibler_divergence_.v_avg kullback_leibler_divergence_avg, 
    kullback_leibler_divergence_.v_stddev kullback_leibler_divergence_stddev,
    network_structure,
    widths
from
(
    SELECT * FROM experiment_ e WHERE
        e.experiment_id IN (
            select distinct r.experiment_id 
            FROM
                run_ r
            WHERE 
                r.record_timestamp >= COALESCE(0, (SELECT MAX(update_timestamp) FROM experiment_summary_))
                AND NOT EXISTS (SELECT * FROM experiment_summary_ s WHERE 
                              s.experiment_id = r.experiment_id 
                              AND s.update_timestamp > r.record_timestamp)
        )
) e,
lateral (
    select
        max(num) num_runs,
        array_agg(num) num,
        array_agg(val_loss_num_finite) val_loss_num_finite,
        array_agg(val_loss_avg) val_loss_avg,
        array_agg(val_loss_stddev) val_loss_stddev,
        array_agg(val_loss_min) val_loss_min,
        array_agg(val_loss_max) val_loss_max,
        array_agg(val_loss_percentile) val_loss_percentile
    from (
        select 
            (COUNT(COALESCE(v, 'NaN'::real)))::smallint num,
            (COUNT(v))::smallint val_loss_num_finite,
            (AVG(v))::real val_loss_avg,
            (stddev_samp(v))::real val_loss_stddev,
            MIN(v) val_loss_min,
            MAX(v) val_loss_max,
            (PERCENTILE_DISC(array[.166, .333, .5, .667, .834]) WITHIN GROUP(ORDER BY  COALESCE(v, 'NaN'::real))) val_loss_percentile
        from
            run_ r,
            lateral unnest(r.val_loss) WITH ORDINALITY as epoch_value(v, epoch)
        where r.experiment_id = e.experiment_id
        group by epoch
        order by epoch
    ) x
) val_loss_,
lateral
(
    select
        array_agg(loss_num_finite) loss_num_finite,
        array_agg(loss_avg) loss_avg,
        array_agg(loss_stddev) loss_stddev,
        array_agg(loss_min) loss_min,
        array_agg(loss_max) loss_max,
        array_agg(loss_percentile) loss_percentile
    from (
    select 
        (COUNT(v))::smallint loss_num_finite,
        (AVG(v))::real loss_avg,
        (stddev_samp(v))::real loss_stddev,
        MIN(v) loss_min,
        MAX(v) loss_max,
        (PERCENTILE_DISC(array[.166, .333, .5, .667, .834]) WITHIN GROUP(ORDER BY  COALESCE(v, 'NaN'::real))) loss_percentile
    from
        run_ r,
        lateral unnest(r.loss) WITH ORDINALITY as epoch_value(v, epoch)
    where r.experiment_id = e.experiment_id
    group by epoch
    order by epoch
    ) x
) loss_,
lateral (
    select array_agg(v_avg) v_avg, array_agg(v_stddev) v_stddev
    from (
      select (AVG(v))::real v_avg,(stddev_samp(v))::real v_stddev
      from run_ r, lateral unnest(r.val_accuracy) WITH ORDINALITY as epoch_value(v, epoch)
      where r.experiment_id = e.experiment_id
      group by epoch order by epoch ) x
) val_accuracy_,
lateral (
    select array_agg(v_avg) v_avg, array_agg(v_stddev) v_stddev
    from (
      select (AVG(v))::real v_avg,(stddev_samp(v))::real v_stddev
      from run_ r, lateral unnest(r.accuracy) WITH ORDINALITY as epoch_value(v, epoch)
      where r.experiment_id = e.experiment_id
      group by epoch order by epoch ) x
) accuracy_,
lateral (
    select array_agg(v_avg) v_avg, array_agg(v_stddev) v_stddev
    from (
      select (AVG(v))::real v_avg,(stddev_samp(v))::real v_stddev
      from run_ r, lateral unnest(r.val_mean_squared_error) WITH ORDINALITY as epoch_value(v, epoch)
      where r.experiment_id = e.experiment_id
      group by epoch order by epoch ) x
) val_mean_squared_error_, 
lateral (
    select array_agg(v_avg) v_avg, array_agg(v_stddev) v_stddev
    from (
      select (AVG(v))::real v_avg,(stddev_samp(v))::real v_stddev
      from run_ r, lateral unnest(r.mean_squared_error) WITH ORDINALITY as epoch_value(v, epoch)
      where r.experiment_id = e.experiment_id
      group by epoch order by epoch ) x
) mean_squared_error_,
lateral (
    select array_agg(v_avg) v_avg, array_agg(v_stddev) v_stddev
    from (
      select (AVG(v))::real v_avg,(stddev_samp(v))::real v_stddev
      from run_ r, lateral unnest(r.val_kullback_leibler_divergence) WITH ORDINALITY as epoch_value(v, epoch)
      where r.experiment_id = e.experiment_id
      group by epoch order by epoch ) x
) val_kullback_leibler_divergence_,
lateral (
    select array_agg(v_avg) v_avg, array_agg(v_stddev) v_stddev
    from (
      select (AVG(v))::real v_avg,(stddev_samp(v))::real v_stddev
      from run_ r, lateral unnest(r.kullback_leibler_divergence) WITH ORDINALITY as epoch_value(v, epoch)
      where r.experiment_id = e.experiment_id
      group by epoch order by epoch ) x
) kullback_leibler_divergence_
ON CONFLICT (experiment_id) DO UPDATE SET
        num_runs = EXCLUDED.num_runs,
        num = EXCLUDED.num,
        val_loss_num_finite = EXCLUDED.val_loss_num_finite,
        val_loss_avg = EXCLUDED.val_loss_avg,
        val_loss_stddev = EXCLUDED.val_loss_stddev,
        val_loss_min = EXCLUDED.val_loss_min,
        val_loss_max = EXCLUDED.val_loss_max,
        val_loss_percentile = EXCLUDED.val_loss_percentile,
        loss_num_finite = EXCLUDED.loss_num_finite,
        loss_avg = EXCLUDED.loss_avg,
        loss_stddev = EXCLUDED.loss_stddev,
        loss_min = EXCLUDED.loss_min,
        loss_max = EXCLUDED.loss_max,
        loss_percentile = EXCLUDED.loss_percentile,
        val_accuracy_avg = EXCLUDED.val_accuracy_avg,
        val_accuracy_stddev = EXCLUDED.val_accuracy_stddev,
        accuracy_avg = EXCLUDED.accuracy_avg,
        accuracy_stddev = EXCLUDED.accuracy_stddev,
        val_mean_squared_error_avg = EXCLUDED.val_mean_squared_error_avg,
        val_mean_squared_error_stddev = EXCLUDED.val_mean_squared_error_stddev,
        mean_squared_error_avg = EXCLUDED.mean_squared_error_avg,
        mean_squared_error_stddev = EXCLUDED.mean_squared_error_stddev,
        val_kullback_leibler_divergence_avg = EXCLUDED.val_kullback_leibler_divergence_avg,
        val_kullback_leibler_divergence_stddev = EXCLUDED.val_kullback_leibler_divergence_stddev,
        kullback_leibler_divergence_avg = EXCLUDED.kullback_leibler_divergence_avg,
        kullback_leibler_divergence_stddev = EXCLUDED.kullback_leibler_divergence_stddev
;