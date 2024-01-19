from dataclasses import dataclass, field
import io
from pprint import pprint
from typing import List, Optional, Any, Dict, Tuple

import numpy
from dmp import parquet_util
from dmp.common import KerasConfig
import dmp.keras_interface.model_serialization as model_serialization
from dmp.layer.layer import Layer

from dmp.keras_interface.keras_utils import make_keras_instance, keras_kwcfg
from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.pruning_experiment.pruning_run_spec import (
    IterativePruningConfig,
)
from dmp.task.experiment.training_experiment.run_spec import RunConfig
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.context import Context
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)


@dataclass
class IterativePruningExperiment(TrainingExperiment):
    pruning: PruningConfig

    @property
    def version(self) -> int:
        return 3

    def __call__(
        self,
        context: Context,
        config: IterativePruningConfig,
    ) -> None:
        print(f"********** 1")

        # make sure we are not saving the trained weights in the callback
        # instead, we will always explicitly save them as part of a checkpoint.
        if config.model_saving is None:
            config.model_saving = ModelSavingSpec(
                True,
                True,
                [],
                [],
                0,
                0,
                0,
            )
        else:
            config.model_saving.save_trained_model = False

        # http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf Algorithim 2
        pruning = self.pruning

        self._setup_environment(config)
        dataset, metrics, loss_metric = self._load_and_prepare_dataset()

        # 1: Create a network with randomly initialization W0 ∈ Rd.
        # 2: Initialize pruning mask to m = 1d.
        network = self._make_network(self.model)
        model = self._make_model_from_network(network, metrics)
        epoch_counter, experiment_history = self._try_restore_checkpoint(
            context, config, model
        )

        rewind_point = TrainingExperimentCheckpoint(
            run_id=config.rewind_run_id,
            load_mask=False,
            load_optimizer=pruning.rewind_optimizer,
            epoch=pruning.rewind_epoch,
        )

        print(f"********** 2")

        # 4: for n ∈ {1, . . . , N} do
        first_iteration = True
        while True:
            iteration = epoch_counter.training_epoch.fit_number

            print(f"********** 3 {epoch_counter.training_epoch}")
            if iteration >= pruning.iterations:
                print(f"********** 4")
                break

            print(f"********** 5")
            is_new_iteration = config.prune_first_iteration or not first_iteration
            if is_new_iteration:
                iteration += 1
                print(f"********** 6")

                # 6: Prune the lowest magnitude entries of WT that remain. Let m[i] = 0 if WT [i] is pruned.
                pruning.method.prune(
                    model.network.structure,
                    model.keras_network.layer_to_keras_map,
                )

                # load rewind point
                rewind_point.resume(model)
                print(f"********** 7")

            print(f"********** 8")
            # 3: Train W0 to Wk with noise u ∼ U: Wk = A 0→k (W0, u).
            # 5: Train m ⊙ Wk to m ⊙ WT with noise u ′∼ U:WT = Ak→T(m ⊙ Wk, u′).
            epoch_counter.reset()
            self._fit_model(
                context,
                config,
                self.fit,
                dataset,
                model,
                [epoch_counter, self._make_early_stopping_callback(epoch_counter)],
                experiment_history=experiment_history,
                new_fit_number=is_new_iteration,
                new_seed=(pruning.new_seed and first_iteration),
                epochs=pruning.max_epochs_per_iteration,
            )

            print(f"********** 9")
            # 7: Return Wk, m  (handled by model saving callback)
            # save weights at this point
            print(f"********** 10")
            config.prune_first_iteration = True
            pruning.new_seed = False

            self._save_checkpoint(
                context,
                config,
                dataset,
                network,
                experiment_history,
                model,
                epoch_counter,
                loss_metric,
            )

            first_iteration = False

        print(f"********** 11")
        self._record_completed_run(
            context,
            config,
            dataset,
            model.network,
            experiment_history,
            loss_metric,
        )

        print(f"********** 12")

        try:
            context.update_summary()
        except Exception as e:
            import traceback

            print(
                f'Exception "{e}" while updating summary. Traceback:\n{traceback.format_exc()}'
            )
        print(f"********** 13")
