from typing import Any, Dict, Union, Optional, Tuple, Callable
import tensorflow.keras as keras
from dmp.common import keras_type_key, tensorflow_type_key, tensorflow_config_key
from dmp.task.task_util import make_dispatcher


def keras_to_config(target: Any) -> Dict[str, Any]:
    s = keras.utils.serialize_keras_object(target)
    if isinstance(s, dict):
        return make_keras_config(s[tensorflow_type_key],
                                 s[tensorflow_config_key])
    if isinstance(s, str):
        return make_keras_config(s)
    raise NotImplementedError('Unknown keras serialization format {s}.')


def keras_from_config(target: Union[str, Dict[str, Any]]) -> Any:
    type, params = get_params_and_type_from_keras_config(target)
    return keras.utils.deserialize_keras_object({
        tensorflow_type_key: type,
        tensorflow_config_key: params
    })


def make_keras_config(type_name: str, params: Optional[dict] = None) -> dict:
    if params is None:
        return {keras_type_key: type_name}

    if keras_type_key in params:
        raise KeyError(f'Type key {keras_type_key} shadows a key in params.')
    config = params.copy()
    config[keras_type_key] = type_name
    return config


def get_params_and_type_from_keras_config(
        config: Union[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    if isinstance(config, str):
        return config, {}
    params = config.copy()
    return params.pop(keras_type_key), params


def make_from_config_using_keras_get(
        config: dict,
        keras_get_function: Callable,
        name: str,  # used for exception messages
) -> Any:
    if config is None:
        return None

    type, params = get_params_and_type_from_keras_config(config)
    result = keras_get_function({
        tensorflow_type_key: type,
        tensorflow_config_key: params
    })
    if result is None:
        raise ValueError(f'Unknown {name}, {config}.')


def make_typed_keras_config_factory(
        name: str,  # name of the thing we are making from the config
        type_dispatch_table: Dict[str, Callable],  # factory dispatch table 
) -> Callable:

    dispatch_function = make_dispatcher(name, type_dispatch_table)

    def factory(
        config: Optional[Dict],  # config to use with type key
        *args,  # forwarded args
        **kwargs,
    ):
        if config is None:
            return None

        type, params = get_params_and_type_from_keras_config(config)
        return dispatch_function(type)(*args, **kwargs, **params)

    return factory


# def count_vars_in_keras_model(model: keras.Model, var_getter) -> int:
#     count = 0
#     for var in var_getter(model):
#         acc = 1
#         for dim in var.get_shape():
#             acc *= int(dim)
#         count += acc
#     return count

# def count_trainable_parameters_in_keras_model(model: keras.Model) -> int:
#     return count_vars_in_keras_model(model, lambda m: m.trainable_variables)

# def count_parameters_in_keras_model(model: keras.Model) -> int:
#     return count_vars_in_keras_model(model, lambda m: m.variables)

# def count_non_trainable_parameters_in_keras_model(model: keras.Model) -> int:
#     return count_vars_in_keras_model(model,
#                                      lambda m: m.non_trainable_variables)
