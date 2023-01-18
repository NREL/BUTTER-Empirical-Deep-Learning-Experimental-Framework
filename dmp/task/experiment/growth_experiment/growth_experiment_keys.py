from dmp.task.experiment.training_experiment.training_experiment_keys import TrainingExperimentKeys


class GrowthExperimentKeys(TrainingExperimentKeys):
    free_parameter_count_key: str = 'free_parameter_count'
    model_number: str = 'model_number'
    model_epoch:int = 'model_epoch'
    retained: str = 'retained'
    layer_map_key: str = 'layer_map'
    # scale_key: str = 'scale'
    # growth_points_key: str = 'growth_points'
    # growth_source_key: str = 'growth_source'
    # model_epoch_key: str = 'model_epoch'
    # parent_epoch_key: str = 'parent_epoch'
