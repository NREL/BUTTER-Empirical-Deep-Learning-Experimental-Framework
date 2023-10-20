from dmp.task.experiment.training_experiment.training_experiment_keys import (
    TrainingExperimentKeys,
)


class GrowthExperimentKeys(TrainingExperimentKeys):
    def __init__(self) -> None:
        super().__init__()

        self.layer_map_key: str = "layer_map"
        # scale_key: str = 'scale'
        # growth_points_key: str = 'growth_points'
        # growth_source_key: str = 'growth_source'
        # model_epoch_key: str = 'fit_epoch'
        # parent_epoch_key: str = 'parent_epoch'


keys = GrowthExperimentKeys()
