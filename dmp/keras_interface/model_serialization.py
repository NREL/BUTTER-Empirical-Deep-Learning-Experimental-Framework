from bisect import bisect_left
from tensorflow import keras
import os
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy
from command_line_tools.run_tools import makedir_if_not_exists
import dmp.keras_interface.access_model_parameters as access_model_parameters
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo

from dmp.model.model_info import ModelInfo
import dmp.parquet_util as parquet_util
import pyarrow
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch

from dmp.task.task import Task
import re
import h5py as h5
import hdf5plugin

model_data_dir = os.path.join(os.getcwd(), "model_data")

max_epochs = int(1e9)
max_parameters = int(100e9)

makedir_if_not_exists(model_data_dir)

task_filename = "task.json"
network_filename = "network.json"
parameters_filename = "parameters.pq"
optimizer_filename = "optimizer.pq"

saved_optimizer_members = (
    "_momentums",
    "_velocities",
    "_velocity_hats",
    "momentums",
    "velocities",
    "velocity_hats",
)
# keras_model_dirname = 'keras_model'


def save_model_data(
    id: UUID,
    model: ModelInfo,
    epoch: TrainingEpoch,
    parent_id: Optional[UUID] = None,
) -> int:
    import os

    data_path = get_model_data_path(id)
    if parent_id is not None and not os.path.exists(data_path):
        parent_path = get_model_data_path(parent_id)
        if os.path.exists(parent_path):
            import shutil

            shutil.copy(parent_path, data_path)

    return save_parameters(
        model.network.structure,
        model.keras_network.layer_to_keras_map,
        model.keras_model.optimizer,
        epoch,
        data_path,
    )


def get_model_data_path(
    run_id: UUID,
) -> str:
    return os.path.join(model_data_dir, f"{str(run_id)}.h5")


def load_model_from_file(
    run_id: UUID,
    epoch: TrainingEpoch,
    model: ModelInfo,
    load_mask: bool = True,
    load_optimizer: bool = True,
) -> TrainingEpoch:
    """
    Loads the parameters for a given training epoch into an existing model.
    """
    data_path = get_model_data_path(run_id)
    return load_parameters(
        model.network.structure,
        model.keras_network.layer_to_keras_map,
        model.keras_model.optimizer if load_optimizer else None,
        data_path,
        epoch,
        load_mask=load_mask,
    )


def find_sequence_number(
    epoch_dataset,
    epoch: TrainingEpoch,
):
    """
    Finds the sequence number cooresponding to the given training epoch.
    Returns None if not found.
    """
    shape = epoch_dataset.shape
    if shape[0] == 0 or shape[1] == 0:
        return None

    # if "sequence_index" not in h5_file:

    sequence_number = epoch_dataset.shape[1] - 1
    while sequence_number >= 0:
        if (
            epoch_dataset[0, sequence_number] == epoch.epoch
            and epoch_dataset[1, sequence_number] == epoch.fit_number
            and epoch_dataset[2, sequence_number] == epoch.fit_epoch
        ):
            return sequence_number

        sequence_number -= 1

    return None


def load_parameters(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    optimizer: Optional[keras.optimizers.Optimizer],
    path: str,
    epoch: TrainingEpoch,
    load_mask: bool = True,
) -> TrainingEpoch:
    with h5.File(path, "r") as h5_file:
        (
            epoch_dataset,
            parameter_dataset,
            optimizer_datasets,
        ) = get_datasets_from_model_file(h5_file, optimizer)

        sequence_number = epoch.sequence_number
        if sequence_number is None:
            sequence_number = find_sequence_number(epoch_dataset, epoch)
            if sequence_number is None:
                raise ValueError(
                    f"Load parameters could not find a matching epoch for {epoch}."
                )

        print(
            f"Loading model with optimizer type: {type(optimizer)} with members {[m[0] for m in optimizer_datasets]}."
        )

        epoch.marker = epoch_dataset[3, sequence_number]
        epoch.sequence_number = sequence_number

        parameter_index = 0

        def visit_variable(layer, keras_layer, i, variable):
            nonlocal parameter_index

            shape = variable.value().shape
            size = numpy.prod(shape)
            parameter_limit = parameter_index + size
            # print(f"loading variable: {variable.name} {size} {shape} {row_index}")
            constraint = access_model_parameters.get_mask_constraint(
                keras_layer, variable
            )

            chunk = parameter_dataset[parameter_index:parameter_limit, sequence_number]
            mask = numpy.logical_not(numpy.isnan(chunk))

            if load_mask and constraint is not None:
                constraint.mask.assign(mask.reshape(shape))

            def load_variable(dataset, variable):
                prepared = dataset[parameter_index:parameter_limit, sequence_number]
                prepared = numpy.where(
                    numpy.logical_and(mask, numpy.logical_not(numpy.isnan(prepared))),
                    prepared,
                    variable.numpy().reshape(mask.shape),
                )
                # print(f"{name}, {row_index}, {size}, {shape}, values: {prepared[0:4]}")
                prepared = prepared.reshape(shape)
                variable.assign(prepared)

            load_variable(parameter_dataset, variable)
            for optimizer_member, member_dataset in optimizer_datasets:
                var_key = optimizer._var_key(variable)
                if var_key not in optimizer._index_dict:
                    continue
                variable_index = optimizer._index_dict[var_key]  # type: ignore
                optimizer_variable = getattr(optimizer, optimizer_member)[
                    variable_index
                ]
                load_variable(member_dataset, optimizer_variable)

            parameter_index = parameter_limit

        access_model_parameters.visit_parameters(
            root,
            layer_to_keras_map,
            visit_variable,
        )

    return epoch


def require_parameter_dataset(
    dest,
    name: str,
    dtype,
):
    return dest.require_dataset(
        name,
        (0, 0),
        dtype=dtype,
        shuffle=False,
        chunks=(256, 128),
        maxshape=(max_parameters, max_epochs),
        fillvalue=numpy.nan,
        **hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
    )


def get_datasets_from_model_file(
    h5_file: h5.File,
    optimizer: Optional[keras.optimizers.Optimizer],
) -> Tuple[h5.Dataset, h5.Dataset, List[Tuple[str, h5.Dataset,]],]:
    """
    "parameter" dataset:
    + every checkpoint saves a new [:, sequence] array with all model parameters
    [parameter index, sequence] = parameter value at sequence number (see "epoch" dataset to map sequence to epoch)
    """
    parameter_dataset = require_parameter_dataset(h5_file, "parameter", numpy.float32)

    """
    "epoch" dataset:
        -> maps to TrainingEpoch fields
        [0, sequence] = global epoch number
        [1, sequence] = fit (model) number
        [2, sequence] = fit (model) epoch
        [3, sequence] = marker (0 = normal, 1 = best weights / early stopping)

    To load the parameter values for a particular epoch:
        1) find the sequence number for that epoch using find_sequence_number() and the epoch dataset
        2) use that sequence number to index into the parameter dataset and access the parameter values for that sequence number
            e.x.: parameter_dataset[parameter_index, sequence_number]
    To look at a parameter's history:
        1) parameter_dataset[parameter_index,:] -> all paramer values for every checkpoint
        2)
    """
    epoch_dataset = h5_file.require_dataset(
        "epoch",
        (4, 0),
        dtype=numpy.int32,
        shuffle=False,
        # chunks=(512, 64),
        maxshape=(4, max_epochs),
        fillvalue=-1,
        **hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
    )

    """
    Optimizer state values are stored in the "optimizer_members" group.
    """
    optimizer_members_group = h5_file.require_group("optimizer_members")
    optimizer_datasets = [
        (
            member,
            require_parameter_dataset(optimizer_members_group, member, numpy.float16),
        )
        for member in saved_optimizer_members
        if optimizer is not None and hasattr(optimizer, member)
    ]

    return (
        epoch_dataset,
        parameter_dataset,
        optimizer_datasets,  # type: ignore
    )


def save_parameters(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    optimizer: Optional[keras.optimizers.Optimizer],
    epoch: TrainingEpoch,
    path: str,
) -> int:
    with h5.File(path, "a") as h5_file:
        (
            epoch_dataset,
            parameter_dataset,
            optimizer_datasets,
        ) = get_datasets_from_model_file(h5_file, optimizer)

        sequence_number = find_sequence_number(epoch_dataset, epoch)

        if sequence_number is None:
            sequence_number = epoch_dataset.shape[1]
            print(
                f"seq {sequence_number}, {epoch_dataset.shape}, {parameter_dataset.shape}"
            )

            epoch_dataset.resize((epoch_dataset.shape[0], sequence_number + 1))

            parameter_dataset.resize((parameter_dataset.shape[0], sequence_number + 1))
            for member, dataset in optimizer_datasets + [(None, parameter_dataset)]:
                dataset.resize((dataset.shape[0], sequence_number + 1))

        epoch.sequence_number = sequence_number

        print(
            f"set {sequence_number}, {epoch_dataset.shape}, {parameter_dataset.shape}, {epoch}"
        )
        epoch_dataset[:, sequence_number] = numpy.array(
            [epoch.epoch, epoch.fit_number, epoch.fit_epoch, epoch.marker],
            dtype=numpy.int32,
        )

        print(
            f"Saving model with optimizer type: {type(optimizer)} with members {[m[0] for m in optimizer_datasets]}."
        )

        parameter_index = 0

        def visit_variable(layer, keras_layer, i, variable):
            nonlocal parameter_index

            shape = variable.value().shape
            size = numpy.prod(shape)
            parameter_limit = parameter_index + size
            constraint = access_model_parameters.get_mask_constraint(
                keras_layer, variable
            )
            mask = None if constraint is None else constraint.mask.numpy().flatten()
            # print(f"saving variable: {variable.name} {size} {shape}")

            def accumulate_value(dataset, value):
                if dataset.shape[0] < parameter_limit:
                    dataset.resize((parameter_limit, dataset.shape[1]))
                dataset[parameter_index:parameter_limit, sequence_number] = value

            def accumulate_variable(dataset, variable):
                value = variable.numpy().flatten()
                if mask is not None:
                    value = numpy.where(mask, value, numpy.nan)
                accumulate_value(dataset, value)

            accumulate_variable(parameter_dataset, variable)
            for optimizer_member, member_dataset in optimizer_datasets:
                # print(f"visit_variable {layer} {variable.name} {layer.name} {member}")
                optimizer_member = getattr(optimizer, optimizer_member)
                # print(f"optimizer_member {type(optimizer_member)}")
                var_key = optimizer._var_key(variable)

                if var_key not in optimizer._index_dict:
                    values = numpy.empty(variable.shape)
                    values.fill(numpy.nan)
                    accumulate_value(member_dataset, values)
                else:
                    variable_index = optimizer._index_dict[var_key]  # type: ignore
                    optimizer_variable = optimizer_member[variable_index]
                    accumulate_variable(member_dataset, optimizer_variable)

            parameter_index = parameter_limit

        access_model_parameters.visit_parameters(
            root,
            layer_to_keras_map,
            visit_variable,
        )
    return sequence_number
