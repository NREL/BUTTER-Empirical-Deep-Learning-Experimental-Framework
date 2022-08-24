update experiment_ x
set primary_sweep = (
    (core_shape or secondary_shape)
    and secondary_depth
    and (core_dataset or secondary_dataset or tertiary_dataset)
    and core_epochs
    and core_regularization
    and core_label_noise
    and core_learning_rate
    and core_batch_size
    and core_optimizer
    and batch = 'fixed_3k_1'
),
   "300_epoch_sweep" = (secondary_epochs and batch = 'fixed_300_1'),
   "30k_epoch_sweep" = (tertiary_epochs and batch = 'fixed_30k_1'),
   "learning_rate_sweep" = (
       core_shape 
       and core_dataset
       and (secondary_depth)
       and core_size 
       and core_epochs 
       and core_regularization 
       and core_label_noise 
       and tertiary_learning_rate
       and core_batch_size
       and core_optimizer
   ),
   "label_noise_sweep" = (
       core_shape 
       and core_dataset
       and (secondary_depth)
       and core_size 
       and core_epochs 
       and core_regularization 
       and (core_label_noise or secondary_label_noise)
       and (core_learning_rate or secondary_learning_rate)
       and core_batch_size
       and core_optimizer
       and batch IN ('fixed_3k_1')
   ),
   "batch_size_sweep" = (
       rectangle_shape 
       and (core_dataset or secondary_dataset)
       and (core_depth)
       and core_size 
       and core_epochs 
       and core_regularization 
       and (core_label_noise)
       and (core_learning_rate)
       and (secondary_batch_size)
       and core_optimizer
       and batch IN ('fixed_3k_1', 'batch_size_1')
   ),
   "regularization_sweep" = (
       rectangle_shape 
       and (core_dataset or secondary_dataset)
       and (core_depth)
--        and core_size 
       and core_epochs 
       and (core_regularization or secondary_regularization) 
       and (core_label_noise)
       and (core_learning_rate)
       and (core_batch_size)
       and core_optimizer
       and batch IN ('fixed_3k_1', 'l1_group_0', 'l2_group_0')
   ),
   "learning_rate_batch_size_sweep" = (
        rectangle_shape
        and core_epochs
        and (core_label_noise)
        and (core_dataset or secondary_dataset)
        and (reduced_core_depth_2)
        and (core_size)
        and (batch_learning_rate_x_batch_size or (batch = 'fixed_3k_1' and core_regularization))
        and lr_batch_size_learning_rate
        and secondary_batch_size
        and core_optimizer
   ),
   "optimizer_sweep" = (
        rectangle_shape
        and core_epochs
        and core_regularization
        and (core_label_noise)
        and (core_dataset or secondary_dataset or tertiary_dataset)
        and (reduced_core_depth)
        and (core_size)
        and tertiary_learning_rate
        and secondary_batch_size
        and batch in ('fixed_3k_1', 'learning_rate', 'batch_size', 'optimizer_1')
    ),
   "size_adjusted_regularization_sweep" = 
        (rectangle_shape
            and core_depth
            and (core_dataset or secondary_dataset)
            and core_epochs
            and core_label_noise
            and core_learning_rate
            and core_batch_size
            and core_optimizer
            and (batch_size_adjusted_regularization_sweep 
            OR (
                batch = 'fixed_3k_1'
                and core_regularization
                )))
        
from
(
    select 
        e.ctid experiment_ctid,
    "batch".string_value batch,
    (shape.string_value = 'rectangle') rectangle_shape,
        (shape.string_value in (
    'rectangle'
    ,'trapezoid'
    ,'exponential'
    ,'wide_first_2x')) core_shape,
        (shape.string_value in (
    'rectangle_residual'
    ,'wide_first_4x'
    ,'wide_first_8x'
    ,'wide_first_16x')) secondary_shape,
         (dataset.string_value in (
            '529_pollen'
            ,'connect_4'
            ,'537_houses'
            ,'mnist'
            ,'201_pol'
            ,'sleep'
            ,'wine_quality_white'
        )) core_dataset,
    (dataset.string_value in (
            'nursery'
            ,'adult'
        )) secondary_dataset,
    (dataset.string_value in (
            '505_tecator'
            ,'294_satellite_images'
        )) tertiary_dataset,
        ("depth".integer_value in (2,3,4,5,6)) reduced_core_depth,
        ("depth".integer_value in (2,3,4,5,6,7)) reduced_core_depth_2,
     ("depth".integer_value in (2,3,4,5,6,7,8,9,10)) core_depth,
    ("depth".integer_value in (2,3,4,5,7,8,9,10,12,14,16,18,20)) secondary_depth,
    ("size".integer_value in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                     32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                     8388608, 16777216)) core_size,
    ("size".integer_value in (33554432, 67108864, 134217728)) "300_size",
    ("size".integer_value in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                     32768, 65536, 131072, 262144)) "30k_size",
    ("kernel_regularizer.type".string_value is NULL) core_regularization,
    ("kernel_regularizer.type".string_value is not NULL) secondary_regularization,
    ("label_noise".real_value = 0.0::real) core_label_noise,
    ("label_noise".real_value > 0.0::real) secondary_label_noise,
    ("batch_size".integer_value = 256) core_batch_size,
    ("batch_size".integer_value IN (32, 64, 128, 256, 512, 1024)) secondary_batch_size,
    ("learning_rate".real_value = 0.0001::real) core_learning_rate,
    ("learning_rate".real_value = 0.001::real) secondary_learning_rate,
    ("learning_rate".real_value in (0.01::real, 0.001::real, 0.0001::real, 0.00001::real)) tertiary_learning_rate,
    ("learning_rate".real_value in (2.5e-05::real, 5e-05::real, 0.0001::real, 0.0002::real, 0.0004::real, 0.0008::real)) lr_batch_size_learning_rate,    
    ("epochs".integer_value = 3000) core_epochs,
    ("epochs".integer_value = 300) secondary_epochs,
    ("epochs".integer_value = 30000) tertiary_epochs,
    ("optimizer".string_value in ('adam')) core_optimizer,
    ("optimizer".string_value in ('adam', 'RMSProp', 'SGD')) secondary_optimizer,
    ("batch".string_value = 'fixed_3k_1') fixed_3k_1_batch,
    ("batch".string_value = 'learning_rate') fixed_3k_1_batch,
    ("batch".string_value = 'learning_rate_x_batch_size') batch_learning_rate_x_batch_size,
    ("batch".string_value = 'optimizer_1') batch_optimizer_1,
    ("batch".string_value in ('l2_group_1', 'l1_group_1')) batch_size_adjusted_regularization_sweep,
        e.*
    from
        experiment_ e
        right join parameter_ "batch" on (e.experiment_parameters @> array["batch".id] and "batch".kind = 'batch')
        right join parameter_ "dataset" on (e.experiment_parameters @> array["dataset".id] and "dataset".kind = 'dataset')
        right join parameter_ "learning_rate" on (e.experiment_parameters @> array["learning_rate".id] and "learning_rate".kind = 'learning_rate')
        right join parameter_ "batch_size" on (e.experiment_parameters @> array["batch_size".id] and "batch_size".kind = 'batch_size')
        left join parameter_ "kernel_regularizer.type" on (e.experiment_parameters @> array["kernel_regularizer.type".id] and "kernel_regularizer.type".kind = 'kernel_regularizer.type')
        left join parameter_ "kernel_regularizer.l1" on (e.experiment_parameters @> array["kernel_regularizer.l1".id] and "kernel_regularizer.l1".kind = 'kernel_regularizer.l1')
        left join parameter_ "kernel_regularizer.l2" on (e.experiment_parameters @> array["kernel_regularizer.l2".id] and "kernel_regularizer.l2".kind = 'kernel_regularizer.l2')
        right join parameter_ "label_noise" on (e.experiment_parameters @> array["label_noise".id] and "label_noise".kind = 'label_noise')
        right join parameter_ "epochs" on (e.experiment_parameters @> array["epochs".id] and "epochs".kind = 'epochs')
        right join parameter_ "shape" on (e.experiment_parameters @> array["shape".id] and "shape".kind = 'shape')
        right join parameter_ "depth" on (e.experiment_parameters @> array["depth".id] and "depth".kind = 'depth')
        right join parameter_ "size" on (e.experiment_parameters @> array["size".id] and "size".kind = 'size')
        right join parameter_ "optimizer" on (e.experiment_parameters @> array["optimizer".id] and "optimizer".kind = 'optimizer')
    where 
        batch.string_value in (
            'fixed_3k_1',
            'batch_size_1',
            'l1_group_0',
            'l2_group_0',
            'fixed_300_1',
            'fixed_30k_1',
            'learning_rate_x_batch_size',
            'optimizer_1',
            'l2_group_1',
            'l1_group_1'
        )
) e
where
    x.ctid = e.experiment_ctid
;

update experiment_ set butter = ("300_epoch_sweep" OR "30k_epoch_sweep" OR batch_size_sweep OR label_noise_sweep OR learning_rate_sweep OR optimizer_sweep OR primary_sweep OR regularization_sweep);

update experiment_summary_ s set 
    "butter" = e."butter",
    "primary_sweep" = e."primary_sweep",
    "300_epoch_sweep" = e."300_epoch_sweep",
    "30k_epoch_sweep" = e."30k_epoch_sweep",
    "learning_rate_sweep" = e."learning_rate_sweep",
    "label_noise_sweep" = e."label_noise_sweep",
    "batch_size_sweep" = e."batch_size_sweep",
    "regularization_sweep" = e."regularization_sweep",
    "learning_rate_batch_size_sweep" = e."learning_rate_batch_size_sweep",
    "optimizer_sweep" = e."optimizer_sweep",
    "size_adjusted_regularization_sweep" = e."size_adjusted_regularization_sweep"
from experiment_ e
where e.experiment_id = s.experiment_id
AND
(  
    s.butter <> e.butter OR
    s.primary_sweep <> e.primary_sweep OR
    s."300_epoch_sweep" <> e."300_epoch_sweep" OR
    s."30k_epoch_sweep" <> e."30k_epoch_sweep" OR
    s.learning_rate_sweep <> e.learning_rate_sweep OR
    s.label_noise_sweep <> e.label_noise_sweep OR
    s.batch_size_sweep <> e.batch_size_sweep OR
    s.regularization_sweep <> e.regularization_sweep OR
    s.learning_rate_batch_size_sweep <> e.learning_rate_batch_size_sweep OR
    s.optimizer_sweep <> e.optimizer_sweep OR
    s.size_adjusted_regularization_sweep <> e.size_adjusted_regularization_sweep
    );


    
alter table experiment_ add column butter boolean default False;
alter table experiment_ add column "primary_sweep" bool not null default False;
alter table experiment_ add column "300_epoch_sweep" bool not null default False;
alter table experiment_ add column "30k_epoch_sweep" bool not null default False;
alter table experiment_ add column "learning_rate_sweep" bool not null default False;
alter table experiment_ add column "label_noise_sweep" bool not null default False;
alter table experiment_ add column "batch_size_sweep" bool not null default False;
alter table experiment_ add column "regularization_sweep" bool not null default False;
alter table experiment_ add column "learning_rate_batch_size_sweep" bool not null default False;
alter table experiment_ add column "optimizer_sweep" bool not null default False;
alter table experiment_ add column "size_adjusted_regularization_sweep" bool not null default False;



alter table experiment_summary_ add column butter boolean default False;
alter table experiment_summary_ add column "primary_sweep" bool not null default False;
alter table experiment_summary_ add column "300_epoch_sweep" bool not null default False;
alter table experiment_summary_ add column "30k_epoch_sweep" bool not null default False;
alter table experiment_summary_ add column "learning_rate_sweep" bool not null default False;
alter table experiment_summary_ add column "label_noise_sweep" bool not null default False;
alter table experiment_summary_ add column "batch_size_sweep" bool not null default False;
alter table experiment_summary_ add column "regularization_sweep" bool not null default False;
alter table experiment_summary_ add column "learning_rate_batch_size_sweep" bool not null default False;
alter table experiment_summary_ add column "optimizer_sweep" bool not null default False;
alter table experiment_summary_ add column "size_adjusted_regularization_sweep" bool not null default False;


select 
    count(*) num_experiments,
    sum(s.num_runs) num_runs,
    avg(s.num_runs) avg_reps,
--     "batch".string_value "batch",
--     "optimizer".string_value "optimizer",
    count(distinct "optimizer".string_value) num_optimizer,
--     "momentum".string_value "momentum",
    count(distinct "momentum".real_value) num_momentum,
--     count(distinct "batch".string_value) num_batches,
--     "dataset".string_value "dataset",    
--     "learning_rate".real_value "learning_rate",
    count(distinct "learning_rate".real_value) num_learning_rates,
--     "batch_size".integer_value "batch_size",
count(distinct "batch_size".integer_value) num_batch_sizes,
--     "kernel_regularizer.type".string_value "kernel_regularizer.type",
count(distinct "kernel_regularizer.type".string_value) num_regularizers,
--     "kernel_regularizer.l1".real_value "kernel_regularizer.l1",
-- count(distinct "kernel_regularizer.l1".real_value) rum_l1,
--     "kernel_regularizer.l2".real_value "kernel_regularizer.l2",
count(distinct "kernel_regularizer.l2".real_value) rum_l2,
--     "label_noise".real_value "label_noise",
    count(distinct "label_noise".real_value) num_label_noise,
--     "epochs".real_value "epochs",
    count(distinct "epochs".integer_value) num_epochs,
--     "shape".string_value "shape",
    count(distinct "shape".string_value) num_shapes,
--     "size".integer_value "size",
    count(distinct size.integer_value) sizes,
--     "depth".integer_value depth,
    count(distinct depth.integer_value) depths,
    min(size.integer_value) min_size,
    max(size.integer_value) max_size,
    min(depth.integer_value) min_depth,
    max(depth.integer_value) max_depth,
    count(distinct "dataset".string_value) num_datasets
from
    experiment_summary_ s
    left join parameter_ "optimizer" on (s.experiment_parameters @> array["optimizer".id] and "optimizer".kind = 'optimizer')
    left join parameter_ "momentum" on (s.experiment_parameters @> array["optimizer".id] and "momentum".kind = 'optimizer.config.momentum')
    left join parameter_ "batch" on (s.experiment_parameters @> array["batch".id] and "batch".kind = 'batch')
    left join parameter_ "dataset" on (s.experiment_parameters @> array["dataset".id] and "dataset".kind = 'dataset')
    left join parameter_ "learning_rate" on (s.experiment_parameters @> array["learning_rate".id] and "learning_rate".kind = 'learning_rate')
    left join parameter_ "batch_size" on (s.experiment_parameters @> array["batch_size".id] and "batch_size".kind = 'batch_size')
    left join parameter_ "kernel_regularizer.type" on (s.experiment_parameters @> array["kernel_regularizer.type".id] and "kernel_regularizer.type".kind = 'kernel_regularizer.type')
    left join parameter_ "kernel_regularizer.l1" on (s.experiment_parameters @> array["kernel_regularizer.l1".id] and "kernel_regularizer.l1".kind = 'kernel_regularizer.l1')
    left join parameter_ "kernel_regularizer.l2" on (s.experiment_parameters @> array["kernel_regularizer.l2".id] and "kernel_regularizer.l2".kind = 'kernel_regularizer.l2')
    left join parameter_ "label_noise" on (s.experiment_parameters @> array["label_noise".id] and "label_noise".kind = 'label_noise')
    left join parameter_ "epochs" on (s.experiment_parameters @> array["epochs".id] and "epochs".kind = 'epochs')
    left join parameter_ "shape" on (s.experiment_parameters @> array["shape".id] and "shape".kind = 'shape')
    left join parameter_ "depth" on (s.experiment_parameters @> array["depth".id] and "depth".kind = 'depth')
    left join parameter_ "size" on (s.experiment_parameters @> array["size".id] and "size".kind = 'size')
where
--     shape.string_value IN (
--         'rectangle'
--         ,'trapezoid'
--         ,'exponential'
--         ,'wide_first_2x'
-- --        ,'wide_first_16x'
-- --        ,'rectangle_residual'
--     )
--     and
--     batch.string_value IN (
--         'optimizer_1'
-- --         'fixed_300_1'
-- --         'fixed_3k_1'
-- --         ,'batch_size_1'
-- --         ,'l1_group_0'
-- --         ,'l2_group_0'
--     )
--     and label_noise.real_value = 0.0::real
--     and learning_rate.real_value = 0.0001::real
--     and batch_size.integer_value = 256
--        "30k_epoch_sweep"
        optimizer_sweep
-- group by 
--     "batch"
--     ,"dataset"
--     "optimizer"
--     ,"momentum"
--     ,"learning_rate"
--     ,"batch_size"
--     ,"kernel_regularizer.type"
--     ,"kernel_regularizer.l1"
--     ,"kernel_regularizer.l2"
--     ,"label_noise"
--     ,"epochs"
--     "shape"
--     ,"size".integer_value
--     ,"depth"
order by
--     "batch"
--     "optimizer"
--     ,"momentum"
--     ,"dataset"
--     ,"learning_rate"
--     ,"batch_size"
--     ,"kernel_regularizer.type"
--     ,"kernel_regularizer.l1"
--     ,"kernel_regularizer.l2"
--     ,"label_noise"
--     ,"epochs"
--     ,"shape"
--     ,"depth"
--     ,"size".integer_value
    num_experiments desc
;