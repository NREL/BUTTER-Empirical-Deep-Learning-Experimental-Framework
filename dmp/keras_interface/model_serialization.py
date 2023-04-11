import os
from typing import Optional, Tuple
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.model.keras_network_info import KerasNetworkInfo

from dmp.model.model_info import ModelInfo

model_data_path = os.path.join(os.getcwd(), 'model_data')
keras_model_dirname = 'keras_model'
network_filename = 'network.json'

class ModelSeralizer:
    def store_model_data(
        self,
        model_info: ModelInfo,
        relative_path: str,
        use_mask: bool = True,
        save_optimizer: bool = True,
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
            model_path,
            network_path,
            keras_model_path
        ) = self.get_paths(relative_path)

        os.makedirs(model_path, exist_ok=True)
        with open(network_path, 'w') as fp:
            simplejson.dump(marshal.marshal(model_info.network), fp)
        model_info.keras_model.save(keras_model_path)

    def load_model_data(
        self,
        relative_path: str,
        target: Optional[ModelInfo] = None,
        use_mask: bool = True,
        save_optimizer: bool = True,
    ) -> ModelInfo:
        
        from dmp.marshaling import marshal
        import simplejson

        (
            model_path,
            network_path,
            keras_model_path
        ) = self.get_paths(relative_path)
        network = None
        with open(network_path, 'r') as fp:
            network = marshal.demarshal(simplejson.load(fp))
        
        from tensorflow import keras
        keras_model = keras.models.load_model(keras_model_path)
        
        # reconstruct Layer to keras.Layer mapping:
        keras_network_info = KerasNetworkInfo({}, [], [])
        
        layer_to_keras_map = keras_network_info.layer_to_keras_map
        for layer in network.layers:
            name = layer.name
            keras_layer = network.get_layer(name)
            layer_to_keras_map[layer] = KerasLayerInfo(layer, keras_layer, keras_layer.output)
        keras_network_info.inputs = keras_model.get_inputs()
        keras_network_info.outputs = keras_model.get_outputs()

        return ModelInfo(network, keras_network_info, keras_model)

    def get_paths(
            self,
            relative_path:str,
    )->Tuple[str, str, str]:
        model_path = os.path.join(model_data_path, relative_path)
        network_path = os.path.join(model_path, network_filename)
        keras_model_path = os.path.join(model_path, keras_model_dirname)
        return (
            model_path,
            network_path,
            keras_model_path
        )

