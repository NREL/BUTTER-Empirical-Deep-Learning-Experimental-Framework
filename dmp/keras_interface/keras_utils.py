from typing import Any, Dict, Union, Optional, Tuple, Callable
import tensorflow.keras as keras
from dmp.common import (
    KerasConfig,
    dispatch,
    keras_type_key,
    tensorflow_type_key,
    tensorflow_config_key,
)


def keras_to_config(target: Any) -> Optional[KerasConfig]:
    '''
    Marshals a keras object into a configuration dictionary.
    The config format is the kwargs dictionary plus an additional key,
    'class_name', with value equal to the classname.
    '''

    if target is None:
        return None

    s = keras.utils.serialize_keras_object(target)
    if isinstance(s, dict):
        return make_keras_config(s[tensorflow_type_key], s[tensorflow_config_key])
    if isinstance(s, str):
        return make_keras_config(s)
    raise NotImplementedError('Unknown keras serialization format {s}.')


def register_custom_keras_type(type_name: str, factory: Callable) -> None:
    '''
    Registers a keras type for use in the configuration marshaling protocol defined in this file.
    '''
    __keras_dispatch_table[type_name] = factory


def register_custom_keras_types(type_map: Dict[str, Callable]) -> None:
    '''
    Registers multiple keras type for use in the configuration marshaling protocol defined in this file.
    '''
    for k, v in type_map.items():
        register_custom_keras_type(k, v)


def make_keras_instance(
    config: Optional[KerasConfig],
    *params,
    **overrides,
) -> Any:
    '''
    Makes a keras instance from a config, as defined in keras_to_config().

    Configs are the kwargs passed to the class constructor plus an additional
    key, 'class_name', with value equal to the classname.
    '''

    if config is None:
        return None

    

    type_name, kwargs = __get_params_and_type_from_keras_config(config)

    factory = dispatch('keras type', __keras_dispatch_table, type_name)
    kwargs.update(overrides)

    return factory(*params, **kwargs)


def make_keras_config(
    type_name: str,  # the name of the keras class
    params: Optional[KerasConfig] = None,  # the args to pass to the keras constructor
) -> KerasConfig:
    '''
    Makes a configuration dictionary that can be turned into a keras instance
    using make_keras_instance.

    Configs are the kwargs passed to the class constructor plus an additional
    key, 'class_name', with value equal to the classname.
    '''

    if params is None:
        return {keras_type_key: type_name}

    if keras_type_key in params:
        raise KeyError(f'Type key {keras_type_key} shadows a key in params.')
    config = params.copy()
    config[keras_type_key] = type_name
    return config


def make_keras_kwcfg(type_name: str, **kwargs) -> KerasConfig:
    '''
    Makes a configuration dictionary that can be turned into a keras instance
    using make_keras_instance.

    Configs are the kwargs passed to the class constructor plus an additional
    key, 'class_name', with value equal to the classname.
    '''

    return make_keras_config(type_name, kwargs)


def __make_keras_dispatch_table() -> Dict[str, Callable]:
    source_modules = (
        keras.optimizers.schedules,
        keras.layers,
        keras.constraints,
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
            if name.startswith('_'):
                continue
            dispatch_table[name] = cls

    # special provision for activation functions...
    for name, c in keras.activations.__dict__.items():
        if name.startswith('_'):
            continue

        def func_outer(name=name, c=c):
            def func(**kwargs):
                return lambda x: c(x, **kwargs)

            return func

        dispatch_table[name] = func_outer()

    return dispatch_table


__keras_dispatch_table: Dict[str, Callable] = __make_keras_dispatch_table()


def __get_params_and_type_from_keras_config(
    config: Union[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    if isinstance(config, str):
        return config, {}
    
    params = {}
    for k, v in config.items():
        if isinstance(v, dict) and keras_type_key in v:
            v = make_keras_instance(v)
        params[k] = v

    return params.pop(keras_type_key), params
