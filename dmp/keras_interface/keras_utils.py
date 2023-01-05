from typing import Any, Dict, Union, Optional, Tuple, Callable
import tensorflow.keras as keras
from dmp.common import dispatch, keras_type_key, tensorflow_type_key, tensorflow_config_key


def keras_to_config(target: Any) -> Dict[str, Any]:
    s = keras.utils.serialize_keras_object(target)
    if isinstance(s, dict):
        return make_keras_config(s[tensorflow_type_key],
                                 s[tensorflow_config_key])
    if isinstance(s, str):
        return make_keras_config(s)
    raise NotImplementedError('Unknown keras serialization format {s}.')


def register_custom_keras_type(type_name: str, factory: Callable) -> None:
    __keras_dispatch_table[type_name] = factory


def register_custom_keras_types(type_map: Dict[str, Callable]) -> None:
    for k, v in type_map.items():
        register_custom_keras_type(k, v)


# __get_keras_factory = make_dispatcher('keras type', __keras_dispatch_table)


def make_keras_instance(
    config: Optional[Dict[str, Any]],
    *params,
    **overrides,
) -> Any:
    if config is None:
        return None

    type_name, kwargs = __get_params_and_type_from_keras_config(config)
    factory = dispatch('keras type', __keras_dispatch_table, type_name)
    kwargs.update(overrides)
    return factory(*params, **kwargs)


def make_keras_config(type_name: str, params: Optional[dict] = None) -> dict:
    if params is None:
        return {keras_type_key: type_name}

    if keras_type_key in params:
        raise KeyError(f'Type key {keras_type_key} shadows a key in params.')
    config = params.copy()
    config[keras_type_key] = type_name
    return config


def make_keras_kwcfg(type_name: str, **kwargs) -> dict:
    return make_keras_config(type_name, kwargs)


def __make_keras_dispatch_table() -> Dict[str, Callable]:
    source_modules = (
        keras.layers,
        keras.regularizers,
        keras.callbacks,
        keras.constraints,
        keras.metrics,
        keras.initializers,
        keras.optimizers,
        keras.losses,
    )  # later modules override/shadow earlier modules

    dispatch_table: Dict[str, Callable] = {}
    for module in source_modules:
        for name, cls in module.__dict__.items():
            dispatch_table[name] = cls

    # special provision for activation functions...
    for name, cls in keras.activations.__dict__.items():
        dispatch_table[name] = lambda *params, **kwargs: (lambda x: cls(
            x, *params, **kwargs))

    return dispatch_table


__keras_dispatch_table: Dict[str, Callable] = __make_keras_dispatch_table()


def __get_params_and_type_from_keras_config(
    config: Union[str, Dict[str, Any]], ) -> Tuple[str, Dict[str, Any]]:
    if isinstance(config, str):
        return config, {}
    params = config.copy()
    return params.pop(keras_type_key), params
