from dmp.task.experiment.training_experiment.training_experiment_keys import TrainingExperimentKeys


class GrowthExperimentKeys(TrainingExperimentKeys):

    def __init__(self) -> None:
        super().__init__()
        self.free_parameter_count_key: str = 'free_parameter_count'
        self.model_number: str = 'model_number'
        self.model_epoch:int = 'model_epoch'
        self.retained: str = 'retained'
        self.layer_map_key: str = 'layer_map'
        # scale_key: str = 'scale'
        # growth_points_key: str = 'growth_points'
        # growth_source_key: str = 'growth_source'
        # model_epoch_key: str = 'model_epoch'
        # parent_epoch_key: str = 'parent_epoch'

keys = GrowthExperimentKeys()