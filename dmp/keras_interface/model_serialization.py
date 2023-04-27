import os
from typing import Optional, Tuple
from dmp.layer.input import Input
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.model.keras_network_info import KerasNetworkInfo

from dmp.model.model_info import ModelInfo
from dmp.model.network_info import NetworkInfo
from dmp.task.experiment.pruning_experiment.weight_mask import WeightMask

model_data_path = os.path.join(os.getcwd(), 'model_data')
keras_model_dirname = 'keras_model'
network_filename = 'network.json'

from tensorflow import keras


class ModelSeralizer:
    def store_model_data(
        self,
        model_info: ModelInfo,
        relative_path: str,
        store_weights_separately: bool = True,
        use_weight_mask: bool = True,
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

        (model_path, network_path, keras_model_path) = self.get_paths(relative_path)

        print(f'1 {relative_path} {model_path} {network_path} {keras_model_path}')
        os.makedirs(model_path, exist_ok=True)
        with open(network_path, 'w') as fp:
            simplejson.dump(marshal.marshal(model_info.network), fp)
        # model_info.keras_model.save(
        #     keras_model_path,
        #     save_traces=True,
        # )
        keras.models.save_model(
            model_info.keras_model,
            keras_model_path,
            overwrite=True,
            save_traces=True,
            save_format='keras',
        )
        
    def store_model_data(
        self,
        model_info: ModelInfo,
        relative_path: str,
        store_weights_separately: bool = True,
        use_weight_mask: bool = True,
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

        (model_path, network_path, keras_model_path) = self.get_paths(relative_path)

        print(f'1 {relative_path} {model_path} {network_path} {keras_model_path}')
        os.makedirs(model_path, exist_ok=True)
        with open(network_path, 'w') as fp:
            simplejson.dump(marshal.marshal(model_info.network), fp)
        # model_info.keras_model.save(
        #     keras_model_path,
        #     save_traces=True,
        # )
        keras.models.save_model(
            model_info.keras_model,
            keras_model_path,
            overwrite=True,
            save_traces=True,
            save_format='keras',
        )

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
    ) -> Tuple[str, str, str]:
        model_path = os.path.join(model_data_path, relative_path)
        network_path = os.path.join(model_path, network_filename)
        keras_model_path = os.path.join(model_path, keras_model_dirname)
        return (model_path, network_path, keras_model_path)
