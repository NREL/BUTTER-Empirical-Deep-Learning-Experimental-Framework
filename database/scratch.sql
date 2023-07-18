with recursive target as (
select
	r.run_id,
	e.experiment_id,
	r.slurm_job_id,
	r.task_version,
	r.num_nodes,
	r.num_cpus,
	r.num_gpus,
	r.gpu_memory,
	r.host_name,
	r.batch,
	d.command,
	(
		run_data || (
		select
			jsonb_object_agg(
				kind,
				COALESCE(to_jsonb(value_bool), to_jsonb(value_int), to_jsonb(value_float), to_jsonb(value_str), value_json)
			   ) experiment_data
		from attr a
		where
			array[a.attr_id] <@ e.experiment_attrs)
	 ) json_data,
	(
		(
		select
			jsonb_object_agg(
				kind,
				COALESCE(to_jsonb(value_bool), to_jsonb(value_int), to_jsonb(value_float), to_jsonb(value_str), value_json)
			   ) experiment_data
		from attr a
		where
			array[a.attr_id] <@ e.experiment_tags)
	) experiment_tags_json
from
	(
		select *
		from
		experiment
		order by experiment_id
		limit 10
	) e
	inner join run r on (r.experiment_id = e.experiment_id)
	inner join job_data d on (d.id = r.run_id)
where
	jsonb_typeof(d.command) = 'object'
),
flat_command as (
    select distinct
		j.key as key,
		(CASE WHEN jsonb_typeof(j.value) = 'object' THEN j.value ELSE NULL END) as value
    from
		target,
    	jsonb_each(target.command) j
	where
		jsonb_typeof(command) = 'object'
union
    select distinct
		concat(f.key, '_', j.key) as key,
		(CASE WHEN jsonb_typeof(j.value) = 'object' THEN j.value ELSE NULL END) as value
    from
		flat_command f,
    	jsonb_each(f.value) j
    where
		jsonb_typeof(f.value) = 'object'
)
select
	run_id,
	experiment_id,
	(
		command
		|| jsonb_build_object('experiment_tags', experiment_tags_json)
		|| jsonb_build_object(
			'runtime', jsonb_build_object(
				'task_version', task_version,
				'slurm_job_id', slurm_job_id,
				'num_nodes', num_nodes,
				'num_cpus', num_cpus,
				'num_gpus', num_gpus,
				'gpu_memory', gpu_memory,
				'host_name', host_name
			)
			|| (
			select
				jsonb_object_agg(p.key, p.value) filtered_data
			from
				jsonb_each(json_data) p
			where
				not exists (
					select 1
					from flat_command
					where
						flat_command.key = p.key
				)
			)
		)
	) command2
from
	target
;





with recursive target as (
select
	r.run_id,
	e.experiment_id,
	r.slurm_job_id,
	r.task_version,
	r.num_nodes,
	r.num_cpus,
	r.num_gpus,
	r.gpu_memory,
	r.host_name,
	r.batch,
	d.command,
	(
		run_data || (
		select
			jsonb_object_agg(
				kind,
				COALESCE(to_jsonb(value_bool), to_jsonb(value_int), to_jsonb(value_float), to_jsonb(value_str), value_json)
			   ) experiment_data
		from attr a
		where
			array[a.attr_id] <@ e.experiment_attrs)
	 ) json_data,
	(
		(
		select
			jsonb_object_agg(
				kind,
				COALESCE(to_jsonb(value_bool), to_jsonb(value_int), to_jsonb(value_float), to_jsonb(value_str), value_json)
			   ) experiment_data
		from attr a
		where
			array[a.attr_id] <@ e.experiment_tags)
	) experiment_tags_json
from
	(
		select *
		from
		experiment
		order by experiment_id
		limit 10
	) e
	inner join run r on (r.experiment_id = e.experiment_id)
	inner join job_data d on (d.id = r.run_id)
where
	jsonb_typeof(d.command) = 'object'
),
flat_command as (
    select distinct
		target.run_id,
		j.key,
		j.value,
    from
		target,
    	jsonb_each(target.command) j
	where
		jsonb_typeof(command) = 'object'
union
    select distinct
		f.run_id,
		concat(f.key, '_', j.key) as key,
		j.value,
    from
		flat_command f,
    	jsonb_each(f.value) j
    where
		jsonb_typeof(f.value) = 'object'
order by run_id, key
)
select
	run_id,
	experiment_id,
	(
		command
		|| jsonb_build_object('experiment_tags', experiment_tags_json)
		|| jsonb_build_object(
			'runtime', jsonb_build_object(
				'task_version', task_version,
				'slurm_job_id', slurm_job_id,
				'num_nodes', num_nodes,
				'num_cpus', num_cpus,
				'num_gpus', num_gpus,
				'gpu_memory', gpu_memory,
				'host_name', host_name
			)
			|| (
			select
				jsonb_object_agg(p.key, p.value) filtered_data
			from
				jsonb_each(json_data) p
			where
				not exists (
					select 1
					from flat_command
					where
						flat_command.key = p.key
				)
			)
		)
	) command2
from
	target
;