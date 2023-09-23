from tensorflow import keras
import os
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy
import dmp.keras_interface.access_model_parameters as access_model_parameters
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo

from dmp.model.model_info import ModelInfo
import dmp.parquet_util as parquet_util
import pyarrow

from dmp.task.task import Task
import re

model_data_path = os.path.join(os.getcwd(), "model_data")

task_filename = "task.json"
network_filename = "network.json"
parameters_filename = "parameters.pq"
optimizer_filename = "optimizer.pq"

saved_optimizer_members = (
    "_momentums",
    "_velocities",
    "_velocity_hats",
    "momentums",
)
# keras_model_dirname = 'keras_model'


def save_model_data(
    task: Task,
    model: ModelInfo,
    model_path: str,
):
    # print(f'smd: {task}\n\n{model}\n\n{model_path}\n\n')
    """
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
    """

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

    with open(task_path, "w") as task_file:
        # print(f"Writing task to {task_path}...")
        simplejson.dump(marshal.marshal(task), task_file)

    with open(network_path, "w") as network_file:
        # print(f"Writing network to {network_path}...")
        simplejson.dump(marshal.marshal(model.network), network_file)

    with open(parameters_path, "wb") as parameters_file:
        # print(f"Writing parameters to {parameters_file}...")
        save_parameters(
            model.network.structure,
            model.keras_network.layer_to_keras_map,
            None,
            parameters_file,
        )

    with open(optimizer_path, "wb") as optimizer_file:
        # print(f"Writing model state to {optimizer_path}...")
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


def load_model_from_file(
    run_id: UUID,
    model_number: int,
    model_epoch: int,
    model: ModelInfo,
    load_mask: bool = True,
    load_optimizer: bool = True,
) -> None:
    model_path = get_path_for_model_savepoint(
        run_id,
        model_number,
        model_epoch,
    )
    (
        absolute_path,
        task_path,
        network_path,
        parameters_path,
        optimizer_path,
    ) = get_paths(model_path)

    optimizer_path = re.sub(r"(\d+)\.0", r"\1", optimizer_path)

    with open(optimizer_path, "rb") as file:
        return load_model(
            model,
            file,
            load_mask=load_mask,
            load_optimizer=load_optimizer,
        )


def load_model(
    model: ModelInfo,
    file,
    load_mask: bool = True,
    load_optimizer: bool = True,
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
    load_mask: bool = True,
) -> None:
    table = parquet_util.read_parquet_table(file)
    table.sort_by([("sequence", "ascending")])
    parameters_table = {
        column: table[column].to_numpy() for column in table.column_names
    }
    # print(f'first values: {table["value"][0:4]}')
    del table

    optimizer_members = [
        member
        for member in saved_optimizer_members
        if optimizer is not None
        and member in parameters_table
        and hasattr(optimizer, member)
    ]

    # print(
    #     f"Loading model with optimizer type: {type(optimizer)} with members {optimizer_members}."
    # )

    row_index = 0

    def visit_variable(layer, keras_layer, i, variable):
        nonlocal row_index

        shape = variable.value().shape
        size = numpy.prod(shape)
        # print(f"loading variable: {variable.name} {size} {shape} {row_index}")
        constraint = access_model_parameters.get_mask_constraint(keras_layer, variable)
        mask = None
        if load_mask and constraint is not None:
            column = parameters_table["value"]
            chunk = column[row_index : row_index + size]
            # mask = numpy.logical_not(pyarrow.compute.is_null(chunk).to_numpy())
            mask = numpy.logical_not(numpy.isnan(chunk))
            # print(f"load_mask {numpy.sum(mask)} / {numpy.size(mask)}")
            constraint.mask.assign(mask.reshape(shape))

        def load_variable(name, variable):
            column = parameters_table[name]
            prepared = column[row_index : row_index + size]

            if mask is not None:
                prepared = numpy.where(mask, prepared, 0)
            # print(f"{name}, {row_index}, {size}, {shape}, values: {prepared[0:4]}")
            prepared = prepared.reshape(shape)
            variable.assign(prepared)

        load_variable("value", variable)
        if optimizer is not None:
            for member in optimizer_members:
                var_key = optimizer._var_key(variable)
                if var_key not in optimizer._index_dict:
                    continue
                variable_index = optimizer._index_dict[var_key]  # type: ignore
                optimizer_variable = getattr(optimizer, member)[variable_index]
                load_variable(member, optimizer_variable)

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

    # print(
    #     f"Saving model with optimizer type: {type(optimizer)} with members {optimizer_members}."
    # )

    data: dict = {column: [] for column in ["value"] + optimizer_members}
    row_index = 0

    def visit_variable(layer, keras_layer, i, variable):
        nonlocal row_index

        shape = variable.value().shape
        size = numpy.prod(shape)
        constraint = access_model_parameters.get_mask_constraint(keras_layer, variable)
        mask = None if constraint is None else constraint.mask.numpy().flatten()
        # print(f"saving variable: {variable.name} {size} {shape}")

        def accumulate_value(name, value):
            data[name].append(value)

        def accumulate_variable(name, variable):
            value = variable.numpy().flatten()
            if mask is not None:
                value = numpy.where(mask, value, numpy.nan)
            # print(
            #     f"{name}, {row_index}, {len(data[name])}, {size}, {shape}, values: {value[0:4]}"
            # )
            accumulate_value(name, value)

        accumulate_variable("value", variable)
        if optimizer is not None:
            for member in optimizer_members:
                # print(f"visit_variable {layer} {variable.name} {layer.name} {member}")
                optimizer_member = getattr(optimizer, member)
                # print(f"optimizer_member {type(optimizer_member)}")
                var_key = optimizer._var_key(variable)

                if var_key not in optimizer._index_dict:
                    values = numpy.empty(variable.shape)
                    values.fill(numpy.nan)
                    accumulate_value(member, values)
                else:
                    variable_index = optimizer._index_dict[var_key]  # type: ignore
                    optimizer_variable = optimizer_member[variable_index]
                    accumulate_variable(member, optimizer_variable)

        row_index += size

    access_model_parameters.visit_parameters(
        root,
        layer_to_keras_map,
        visit_variable,
    )

    data = {k: numpy.concatenate(v) for k, v in data.items()}
    data["sequence"] = numpy.arange(0, row_index)
    cols = sorted(data.keys())

    table, use_byte_stream_split = parquet_util.make_pyarrow_table_from_numpy(
        cols,
        [data[c] for c in cols],
        nan_to_none=True,
    )

    # print(f'first values: {table["value"][0:4]}, {table["sequence"][0:4]}')

    parquet_util.write_parquet_table(
        table,
        file,
        use_byte_stream_split,
    )


def get_path_for_model_savepoint(
    run_id: UUID,
    model_number: int,
    model_epoch: int,
) -> str:
    return os.path.join(str(run_id), f"{model_number}_{model_epoch}")
