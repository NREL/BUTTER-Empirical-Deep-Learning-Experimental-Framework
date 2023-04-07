from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Optional, Any, Dict, Tuple

from jobqueue.job import Job
import numpy
from dmp import parquet_util
from dmp.common import KerasConfig
from dmp.layer.layer import Layer

from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_kwcfg
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord

from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.keras_interface.access_model_weights import (
    AccessModelWeights,
)
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.worker import Worker


@dataclass
class IterativePruningExperiment(TrainingExperiment):
    num_pruning_iterations: int

    pre_prune_epochs: int
    pre_pruning_trigger: Optional[KerasConfig]

    pruning_method: PruningMethod
    pruning_trigger: Optional[KerasConfig]
    max_pruning_epochs: int

    rewind: bool

    @property
    def version(self) -> int:
        return 0

    def __call__(
        self,
        worker: Worker,
        job: Job,
        *args,
        **kwargs,
    ) -> ExperimentResultRecord:
        # http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf Algorithim 2

        if self.pruning_trigger is None:
            self.pruning_trigger = self.early_stopping

        with worker.strategy.scope():
            # tensorflow.config.optimizer.set_jit(True)
            self._set_random_seeds()
            dataset = self._load_and_prepare_dataset()
            metrics = self._autoconfigure_for_dataset(dataset)

            # 1: Create a network with randomly initialization W0 ∈ Rd.
            # 2: Initialize pruning mask to m = 1d.
            model = self._make_model(worker, self.model)
            self._compile_model(dataset, model, metrics)

            # 3: Train W0 to Wk with noise u ∼ U: Wk = A 0→k (W0, u).
            early_stopping = make_keras_instance(self.pre_pruning_trigger)
            model_history = self._fit_model(
                self.fit,
                dataset,
                model,
                [early_stopping],
                epochs=self.pre_prune_epochs,
            )

            history = {}
            num_free_parameters = model.network.num_free_parameters
            self._accumulate_model_history(
                history,
                model_history,
                num_free_parameters,
                early_stopping,
            )

            # save weights at this point for rewinding
            rewind_weights = {}
            if self.rewind:
                rewind_weights = AccessModelWeights.get_weights(
                    model.network.structure,
                    model.keras_network.layer_to_keras_map,
                    use_mask=False,
                )

            self.compress_weights(model.network.structure, num_free_parameters, rewind_weights)

            # 4: for n ∈ {1, . . . , N} do
            for iteration_n in range(self.num_pruning_iterations):
                # 5: Train m ⊙ Wk to m ⊙ WT with noise u ′∼ U:WT = Ak→T(m ⊙ Wk, u′).

                early_stopping = make_keras_instance(self.pruning_trigger)
                model_history = self._fit_model(
                    self.fit,
                    dataset,
                    model,
                    [early_stopping],
                    epochs=self.max_pruning_epochs,
                )

                # model.network.num_free_parameters
                self._accumulate_model_history(
                    history,
                    model_history,
                    num_free_parameters,
                    early_stopping,
                )

                # 6: Prune the lowest magnitude entries of WT that remain. Let m[i] = 0 if WT [i] is pruned.
                num_pruned = self.pruning_method.prune(
                    model.network.structure,
                    model.keras_network.layer_to_keras_map,
                )
                num_free_parameters = model.network.num_free_parameters - num_pruned

                self.compress_weights(
                    model.network.structure,
                    num_free_parameters,
                    AccessModelWeights.get_weights(
                        model.network.structure,
                        model.keras_network.layer_to_keras_map,
                        use_mask=True,
                    ),
                )
                if self.rewind:
                    AccessModelWeights.set_weights(
                        model.network.structure,
                        model.keras_network.layer_to_keras_map,
                        rewind_weights,
                        use_mask=False,
                    )

            # 7: Return Wk, m
            return self._make_result_record(
                worker.worker_info,
                job.id,
                dataset,
                model.network,
                history,
            )

    def compress_weights(
        self,
        root: Layer,
        num_free_parameters: int,
        weight_map: Dict[Layer, List[numpy.ndarray]],
    ):
        # weight_shape_col = []
        # weight_values_col = []
        w = []

        num_weights = 0
        for layer in root.all_descendants:
            layer_weights = weight_map.get(layer, None)
            shape = None
            if layer_weights is not None:
                for lw in layer_weights:
                    lw = lw.flatten().astype(numpy.float32)

                    w.append(lw)
                    # shape = weights.shape
                    # weights = weights

                    num_weights += lw.size
            # weight_shape_col.append(shape)
            # weight_values_col.append(weights)
        weight_array = numpy.concatenate(w)
        del w

        table, use_byte_stream_split = parquet_util.make_pyarrow_table_from_numpy(
            # ["shape", "weight"],
            # [weight_shape_col, weight_values_col],
            ['weight'],
            [weight_array],
            # [[None if numpy.isnan(e) else e for e in weight_array]],
            nan_to_none=True,
        )
        # print(table['weight'].dtype)
        # print(table['weight'])

        import pyarrow
        import pyarrow.parquet as parquet
        import io

        buffer = io.BytesIO()
        pyarrow_file = pyarrow.PythonFile(buffer)

        parquet.write_table(
            table,
            pyarrow_file,
            # root_path=dataset_path,
            # schema=schema,
            # partition_cols=partition_cols,
            data_page_size=8 * 1024,
            # compression='BROTLI',
            # compression_level=8,
            compression='ZSTD',
            # compression_level=12,
            compression_level=30,
            use_dictionary=False,
            use_byte_stream_split=use_byte_stream_split,
            version='2.6',
            data_page_version='2.0',
            # existing_data_behavior='overwrite_or_ignore',
            # use_legacy_dataset=False,
            write_statistics=False,
            # write_batch_size=64,
            # dictionary_pagesize_limit=64*1024,
        )
        bytes_written = buffer.getbuffer().nbytes
        print(
            f'Compressed {num_free_parameters}/{num_weights} weights into {bytes_written} bytes {float(bytes_written) / num_weights} per weight, {float(bytes_written) / num_free_parameters} per unmasked weight.'
        )
