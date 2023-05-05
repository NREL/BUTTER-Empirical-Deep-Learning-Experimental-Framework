import os
from typing import Dict, List, Optional, Tuple

import numpy
import dmp.keras_interface.access_model_parameters as access_model_parameters
from dmp.layer.input import Input
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.model.keras_network_info import KerasNetworkInfo

from dmp.model.model_info import ModelInfo
from dmp.model.network_info import NetworkInfo
from dmp.task.experiment.experiment_task import ExperimentTask
import dmp.parquet_util as parquet_util
import pyarrow

model_data_path = os.path.join(os.getcwd(), 'model_data')

task_filename = 'task.json'
network_filename = 'network.json'
parameters_filename = 'parameters.pq'
optimizer_filename = 'optimizer.pq'

saved_optimizer_members = (
    '_momentums',
    '_velocities',
    '_velocity_hats',
    'momentums',
)
# keras_model_dirname = 'keras_model'

from tensorflow import keras


def save_model_data(
    task: ExperimentTask,
    model: ModelInfo,
    model_path: str,
):
    '''
    + save file manifest
        + version [int 32]
        + item type [int 32], item version [int 32], item size [int 64]
    + save serialized layer graph
    + save parameters and masks
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
        parameters_path,
        optimizer_path,
    ) = get_paths(model_path)

    os.makedirs(absolute_path, exist_ok=True)

    # print(f'1 {relative_path} {model_path} {network_path} {keras_model_path}')

    with open(task_path, 'w') as task_file:
        print(f'Writing task to {task_path}...')
        simplejson.dump(marshal.marshal(task), task_file)

    with open(network_path, 'w') as network_file:
        print(f'Writing network to {network_path}...')
        simplejson.dump(marshal.marshal(model.network), network_file)

    with open(parameters_path, 'wb') as parameters_file:
        print(f'Writing parameters to {parameters_file}...')
        save_parameters(
            model.network.structure,
            model.keras_network.layer_to_keras_map,
            None,
            parameters_file,
        )

    with open(optimizer_path, 'wb') as optimizer_file:
        print(f'Writing model state to {optimizer_path}...')
        save_parameters(
            model.network.structure,
            model.keras_network.layer_to_keras_map,
            model.keras_model.optimizer,
            optimizer_file,
        )


def get_paths(
    relative_path: str,
) -> Tuple[str, str, str, str, str]:
    absolute_path = os.path.join(model_data_path, relative_path)
    task_path = os.path.join(absolute_path, task_filename)
    network_path = os.path.join(absolute_path, network_filename)
    parameters_path = os.path.join(absolute_path, parameters_filename)
    optimizer_path = os.path.join(absolute_path, optimizer_filename)
    return (
        absolute_path,
        task_path,
        network_path,
        parameters_path,
        optimizer_path,
    )


def load_model(
    model: ModelInfo,
    file,
    load_mask:bool = True,
    load_optimizer:bool = True,
):
    load_parameters(
        model.network.structure,
        model.keras_network.layer_to_keras_map,
        model.keras_model.optimizer if load_optimizer else None,
        file,
        load_mask=load_mask,
    )


def save_model(
    model: ModelInfo,
    file,
):
    save_parameters(
        model.network.structure,
        model.keras_network.layer_to_keras_map,
        model.keras_model.optimizer,
        file,
    )


def load_parameters(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    optimizer: Optional[keras.optimizers.Optimizer],
    file,
    load_mask:bool = True,
) -> None:
    parameters_table = parquet_util.read_parquet_table(file)
    columns = set(parameters_table.column_names)

    optimizer_members = [
        member
        for member in saved_optimizer_members
        if optimizer is not None and member in columns and hasattr(optimizer, member)
    ]

    row_index = 0

    def visit_variable(layer, keras_layer, i, variable):
        nonlocal row_index

        size = numpy.prod(variable.value().shape)
        print(f'variable: {variable.name} {size} {variable.value().shape}')
        constraint = access_model_parameters.get_mask_constraint(keras_layer, variable)
        mask = None
        if load_mask and constraint is not None:
            column = parameters_table['value']
            chunk = column[row_index : row_index + size]
            mask = numpy.logical_not(pyarrow.compute.is_null(chunk).to_numpy())
            constraint.mask.assign(mask.reshape(constraint.mask.value().shape))

        def load_value(column, variable):
            column = parameters_table[column]
            chunk = column[row_index : row_index + size]

            prepared = chunk.to_numpy()
            if mask is not None:
                prepared = numpy.where(mask, prepared, 0)
            prepared = prepared.reshape(variable.value().shape)
            variable.assign(prepared)

        load_value('value', variable)
        for member in optimizer_members:
            variable_index = optimizer._index_dict[optimizer._var_key(variable)]  # type: ignore
            load_value(member, getattr(optimizer, member)[variable_index])

        row_index += size

    access_model_parameters.visit_parameters(
        root,
        layer_to_keras_map,
        visit_variable,
    )


def save_parameters(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    optimizer: Optional[keras.optimizers.Optimizer],
    file,
):
    optimizer_members = [
        member
        for member in saved_optimizer_members
        if optimizer is not None and hasattr(optimizer, member)
    ]

    print(f'opt type: {type(optimizer)} with members {optimizer_members}.')

    data: dict = {column: [] for column in ['value'] + optimizer_members}

    def visit_variable(layer, keras_layer, i, variable):
        constraint = access_model_parameters.get_mask_constraint(keras_layer, variable)
        mask = None if constraint is None else constraint.mask.numpy().flatten()

        def accumulate_value(column, variable):
            value = variable.numpy().flatten()
            if mask is not None:
                value = numpy.where(mask, value, numpy.nan)
            data[column].append(value)

        accumulate_value('value', variable)
        for member in optimizer_members:
            optimizer_member = getattr(optimizer, member)
            variable_index = optimizer._index_dict[optimizer._var_key(variable)]  # type: ignore
            accumulate_value(member, optimizer_member[variable_index])

    access_model_parameters.visit_parameters(
        root,
        layer_to_keras_map,
        visit_variable,
    )

    data = {k: numpy.concatenate(v) for k, v in data.items()}
    cols = sorted(data.keys())

    table, use_byte_stream_split = parquet_util.make_pyarrow_table_from_numpy(
        cols,
        [data[c] for c in cols],
        nan_to_none=True,
    )

    parquet_util.write_parquet_table(
        table,
        file,
        use_byte_stream_split,
    )
