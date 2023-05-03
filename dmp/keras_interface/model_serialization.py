import os
from typing import Dict, List, Optional, Tuple

import numpy
import dmp.keras_interface.access_model_weights as access_model_weights
from dmp.layer.input import Input
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.model.keras_network_info import KerasNetworkInfo

from dmp.model.model_info import ModelInfo
from dmp.model.network_info import NetworkInfo
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.task.experiment.pruning_experiment.weight_mask import WeightMask
import dmp.parquet_util as parquet_util

model_data_path = os.path.join(os.getcwd(), 'model_data')

task_filename = 'task.json'
network_filename = 'network.json'
weights_filename = 'weights.pq'
optimizer_filename = 'optimizer.pq'
# keras_model_dirname = 'keras_model'

from tensorflow import keras


class ModelSeralizer:
    def store_model_data(
        self,
        task: ExperimentTask,
        model: ModelInfo,
        optimizer,
        model_path: str,
    ):
        '''
        + save file manifest
            + version [int 32]
            + item type [int 32], item version [int 32], item size [int 64]
        + save serialized layer graph
        + save weights and masks
        + save optimizer details


        + using keras model saving:
            + save layer graph
            + save keras model
            + use layer graph naming scheme to map layers to keras layers
        '''

        from dmp.marshaling import marshal
        import simplejson

        # from pprint import pprint

        (
            absolute_path,
            task_path,
            network_path,
            weights_path,
            optimizer_path,
        ) = self.get_paths(model_path)

        os.makedirs(absolute_path, exist_ok=True)

        # print(f'1 {relative_path} {model_path} {network_path} {keras_model_path}')

        with open(task_path, 'w') as task_file:
            print(f'Writing task to {task_path}...')
            simplejson.dump(marshal.marshal(task), task_file)

        with open(network_path, 'w') as network_file:
            print(f'Writing network to {network_path}...')
            simplejson.dump(marshal.marshal(model.network), network_file)

        with open(weights_path, 'wb') as weights_file:
            print(f'Writing weights to {weights_file}...')
            self.weights_to_file(
                model.network.structure,
                access_model_weights.get_weights(
                    model.network.structure,
                    model.keras_network.layer_to_keras_map,
                    use_mask=True,
                ),
                weights_file,
            )

        with open(optimizer_path, 'wb') as optimizer_file:
            print(f'Writing optimizer state to {optimizer_path}...')
            self.optimizer_to_file(
                model,
                optimizer_file,
            )

        # model_info.keras_model.save(
        #     keras_model_path,
        #     save_traces=True,
        # )
        # keras.models.save_model(
        #     model_info.keras_model,
        #     keras_model_path,
        #     overwrite=True,
        #     save_traces=True,
        #     save_format='keras',
        # )

    # def store_model_data(
    #     self,
    #     model_info: ModelInfo,
    #     relative_path: str,
    #     store_weights_separately: bool = True,
    #     use_weight_mask: bool = True,
    # ):
    #     '''
    #     + save file manifest
    #         + version [int 32]
    #         + item type [int 32], item version [int 32], item size [int 64]
    #     + save serialized layer graph
    #     + save weights and masks
    #     + save optimizer details

    #     + using keras model saving:
    #         + save layer graph
    #         + save keras model
    #         + use layer graph naming scheme to map layers to keras layers
    #     '''

    #     from dmp.marshaling import marshal
    #     import simplejson

    #     # from pprint import pprint

    #     (model_path, network_path, keras_model_path) = self.get_paths(relative_path)

    #     print(f'1 {relative_path} {model_path} {network_path} {keras_model_path}')
    #     os.makedirs(model_path, exist_ok=True)
    #     with open(network_path, 'w') as fp:
    #         simplejson.dump(marshal.marshal(model_info.network), fp)
    #     # model_info.keras_model.save(
    #     #     keras_model_path,
    #     #     save_traces=True,
    #     # )
    #     keras.models.save_model(
    #         model_info.keras_model,
    #         keras_model_path,
    #         overwrite=True,
    #         save_traces=True,
    #         save_format='keras',
    #     )

    def load_model_data(
        self,
        relative_path: str,
        target: Optional[ModelInfo] = None,
        use_mask: bool = True,
        save_optimizer: bool = True,
    ) -> ModelInfo:
        from dmp.marshaling import marshal
        import simplejson

        (model_path, network_path, keras_model_path) = self.get_paths(relative_path)

        with open(network_path, 'r') as fp:
            network: NetworkInfo = marshal.demarshal(simplejson.load(fp))

        from dmp.task.experiment.recorder.timestamp_recorder import TimestampRecorder
        from dmp.task.experiment.recorder.test_set_history_recorder import (
            TestSetHistoryRecorder,
        )
        from dmp.task.experiment.recorder.zero_epoch_recorder import ZeroEpochRecorder

        keras_model = keras.models.load_model(
            keras_model_path,
            compile=True,
            custom_objects={
                "WeightMask": WeightMask,
                'TimestampRecorder': TimestampRecorder,
                'TestSetHistoryRecorder': TestSetHistoryRecorder,
                'ZeroEpochRecorder': ZeroEpochRecorder,
            },
        )

        # reconstruct Layer to keras.Layer mapping:
        keras_network_info = KerasNetworkInfo({}, [], [])

        layer_to_keras_map = keras_network_info.layer_to_keras_map
        for layer in network.structure.layers:
            name = layer.name
            keras_layer = keras_model.get_layer(name)
            layer_to_keras_map[layer] = KerasLayerInfo(
                layer, keras_layer, keras_layer.output
            )

        # NB: currently only works with a single output
        keras_network_info.outputs = [layer_to_keras_map[network.structure].keras_layer]
        keras_network_info.inputs = [
            layer_to_keras_map[layer].keras_layer
            for layer in network.structure.leaves
            if isinstance(layer, Input)
        ]

        return ModelInfo(network, keras_network_info, keras_model)

    def get_paths(
        self,
        relative_path: str,
    ) -> Tuple[str, str, str, str, str]:
        absolute_path = os.path.join(model_data_path, relative_path)
        task_path = os.path.join(absolute_path, task_filename)
        network_path = os.path.join(absolute_path, network_filename)
        weights_path = os.path.join(absolute_path, weights_filename)
        optimizer_path = os.path.join(absolute_path, optimizer_filename)
        return (
            absolute_path,
            task_path,
            network_path,
            weights_path,
            optimizer_path,
        )

    def weights_to_file(
        self,
        root: Layer,
        weight_map: Dict[Layer, List[numpy.ndarray]],
        file,
    ):
        w = []
        num_weights = 0
        for layer in root.layers_pre_ordered:
            layer_weights = weight_map.get(layer, None)
            if layer_weights is not None:
                for lw in layer_weights:
                    lw = lw.flatten().astype(numpy.float32)
                    w.append(lw)
                    num_weights += lw.size
        weight_array = numpy.concatenate(w)
        del w

        table, use_byte_stream_split = parquet_util.make_pyarrow_table_from_numpy(
            # ["shape", "weight"],
            # [weight_shape_col, weight_values_col],
            ['value'],
            [weight_array],
            # [[None if numpy.isnan(e) else e for e in weight_array]],
            nan_to_none=True,
        )

        parquet_util.write_parquet_table(
            table,
            file,
            use_byte_stream_split,
        )

    def optimizer_to_file(
        self,
        model: ModelInfo,
        file,
    ):
        # weight_shape_col = []
        # weight_values_col = []
        # values = {}
        optimizer = model.keras_model.optimizer

        optimizer_members = [
            member
            for member in (
                '_momentums',
                '_velocities',
                '_velocity_hats',
            )
            if hasattr(optimizer, member)
        ]

        print(f'opt type: {type(optimizer)} with members {optimizer_members}.')

        data: dict = {column: [] for column in ['value'] + optimizer_members}

        def visit_variable(layer, keras_layer, i, variable):
            value = variable.numpy()
            constraint = access_model_weights.get_mask_constraint(keras_layer, variable)
            if constraint is not None:
                value = numpy.where(
                    constraint.mask.numpy(),
                    value,
                    numpy.nan,
                )
            value = variable.numpy().astype(numpy.float32).flatten()
            data['value'].append(value)

            mask = None
            constraint = access_model_weights.get_mask_constraint(keras_layer, variable)
            if constraint is not None:
                mask = constraint.mask.numpy().flatten()

            variable_key = optimizer._var_key(variable)
            variable_index = optimizer._index_dict[variable_key]
            for member in optimizer_members:
                member_value = (
                    getattr(optimizer, member)[variable_index]
                    .numpy()
                    .flatten()
                    .astype(numpy.float32)
                )
                if mask is not None:
                    member_value = numpy.where(
                        mask,
                        value,
                        numpy.nan,
                    )
                data[member].append(member_value)

        access_model_weights.visit_weights(
            model.network.structure,
            model.keras_network.layer_to_keras_map,
            visit_variable,
        )

        data = {k: numpy.concatenate(v) for k, v in data.items()}
        cols = sorted(data.keys())

        table, use_byte_stream_split = parquet_util.make_pyarrow_table_from_numpy(
            # ["shape", "weight"],
            # [weight_shape_col, weight_values_col],
            cols,
            [data[c] for c in cols],
            # [[None if numpy.isnan(e) else e for e in weight_array]],
            nan_to_none=True,
        )
        # print(table['value'].dtype)
        # print(table['value'])
        parquet_util.write_parquet_table(
            table,
            file,
            use_byte_stream_split,
        )