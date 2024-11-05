import os
from typing import Dict, List, Optional, Sequence, Sequence, Tuple
from uuid import UUID

import numpy
from tensorflow import keras
import dmp.keras_interface.access_model_parameters as access_model_parameters
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo

from dmp.model.model_info import ModelInfo
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch

import h5py as h5
import hdf5plugin

import dmp.keras_interface.model_serialization_core as model_serialization_core

model_data_dir = model_serialization_core.model_data_dir


saved_optimizer_members = model_serialization_core.saved_optimizer_members
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
        model.keras_model.optimizer,
        epoch,
        data_path,
    )


def get_model_data_path(
    run_id: UUID,
) -> str:
    return model_serialization_core.get_model_data_path(run_id)


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
        model.keras_model.optimizer if load_optimizer else None,
        data_path,
        epoch,
        load_mask=load_mask,
    )


def find_sequence_number(
    epoch_dataset,
    epoch: TrainingEpoch,
):
    return model_serialization_core.find_sequence_number(epoch_dataset, epoch)


def load_parameters(
    root: Layer,
    optimizer: Optional[keras.optimizers.Optimizer],
    path: str,
    epoch: TrainingEpoch,
    load_mask: bool = True,
    preserve_masked_parameters=True,
    and_with_existing_mask=True,
) -> TrainingEpoch:
    with h5.File(path, "r") as h5_file:
        (
            epoch_dataset,
            parameter_dataset,
            optimizer_datasets,
        ) = get_datasets_from_model_file_using_optimizer(h5_file, optimizer)

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

        load_parameters_from_datasets(
            root,
            optimizer,
            load_mask,
            parameter_dataset,
            optimizer_datasets,
            sequence_number,
            preserve_masked_parameters,
            and_with_existing_mask,
        )

    return epoch


def load_parameters_from_datasets(
    root,
    optimizer,
    load_mask,
    parameter_dataset,
    optimizer_datasets,
    sequence_number,
    preserve_masked_parameters=True,
    and_with_existing_mask=True,
):
    parameter_index = 0

    def visit_variable(layer, keras_layer, i, variable):
        nonlocal parameter_index

        shape = variable.value.shape
        size = numpy.prod(shape)
        parameter_limit = parameter_index + size
        print(f"loading variable: {variable.name} {size} {shape} {parameter_index}")
        constraint = access_model_parameters.get_mask_constraint(keras_layer, variable)

        chunk = parameter_dataset[parameter_index:parameter_limit, sequence_number]

        previous_mask = None
        if constraint is not None:
            previous_mask = constraint.get_mask(shape)
        incoming_mask = numpy.logical_not(numpy.isnan(chunk).reshape(shape))
        new_mask = previous_mask

        if load_mask:
            new_mask = incoming_mask
            if and_with_existing_mask and previous_mask is not None:
                new_mask = numpy.logical_and(incoming_mask, previous_mask)

            constraint.set_mask(new_mask)

        def load_variable(dataset, variable):
            prepared = dataset[parameter_index:parameter_limit, sequence_number]
            prepared = prepared.reshape(shape)
            prepared = numpy.where(incoming_mask, prepared, 0)

            if preserve_masked_parameters and new_mask is not None:
                prepared = numpy.where(
                    new_mask,
                    prepared,
                    variable.numpy(),
                )

            # print(f"{name}, {row_index}, {size}, {shape}, values: {prepared[0:4]}")
            variable.assign(prepared)

        load_variable(parameter_dataset, variable)

        for optimizer_member, member_dataset in optimizer_datasets:
            optimizer_variable = get_optimizer_variable(
                optimizer,
                optimizer_member,
                variable,
            )
            print(
                f"load optimizer variable {variable.name} {optimizer_member}, {type(optimizer_variable)}"
            )
            if optimizer_variable is not None:
                load_variable(member_dataset, optimizer_variable)

        parameter_index = parameter_limit

    access_model_parameters.visit_parameters(
        root,
        visit_variable,
    )


def get_optimizer_variable(optimizer, optimizer_member, variable):
    if not hasattr(optimizer, optimizer_member):
        return None
    optimizer_variables = getattr(optimizer, optimizer_member)

    if hasattr(optimizer, "_get_variable_index"):
        try:
            variable_index = optimizer._get_variable_index(variable)
        except KeyError:
            return None
    elif hasattr(optimizer, "_var_key"):
        var_key = optimizer._var_key(variable)
        if var_key not in optimizer._index_dict:
            return None
        variable_index = optimizer._index_dict[var_key]  # type: ignore
    else:
        print("Could not determine how to access optimizer state variables!")
        return None

    if len(optimizer_variables) <= variable_index:
        return None

    print(
        f"try load optimizer variable {variable.name} {optimizer_member}, {type(optimizer_variables)} {len(optimizer_variables)} {variable_index}"
    )
    return optimizer_variables[variable_index]


def require_parameter_dataset(
    dest,
    name: str,
    dtype,
):
    return model_serialization_core.require_parameter_dataset(dest, name, dtype)


def get_datasets_from_model_file_using_optimizer(
    h5_file: h5.File,
    optimizer: Optional[keras.optimizers.Optimizer],
) -> Tuple[
    h5.Dataset,
    h5.Dataset,
    List[
        Tuple[
            str,
            h5.Dataset,
        ]
    ],
]:
    optimizer_members = [
        member
        for member in saved_optimizer_members
        if optimizer is not None and hasattr(optimizer, member)
    ]
    print(f"Found optimizer {type(optimizer)} with members {optimizer_members}.")
    return get_datasets_from_model_file(
        h5_file,
        optimizer_members,
    )


def get_datasets_from_model_file(
    h5_file: h5.File,
    optimizer_members: Optional[Sequence[str]],
) -> Tuple[
    h5.Dataset,
    h5.Dataset,
    List[
        Tuple[
            str,
            h5.Dataset,
        ]
    ],
]:
    return model_serialization_core.get_datasets_from_model_file(
        h5_file, optimizer_members
    )


def convert_epoch_dataset_into_epochs(epoch_dataset):
    return model_serialization_core.convert_epoch_dataset_into_epochs(epoch_dataset)


def save_parameters(
    root: Layer,
    optimizer: Optional[keras.optimizers.Optimizer],
    epoch: TrainingEpoch,
    path: str,
) -> int:
    with h5.File(path, "a") as h5_file:
        (
            epoch_dataset,
            parameter_dataset,
            optimizer_datasets,
        ) = get_datasets_from_model_file_using_optimizer(h5_file, optimizer)

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

            shape = variable.shape
            size = numpy.prod(shape)
            parameter_limit = parameter_index + size
            constraint = access_model_parameters.get_mask_constraint(
                keras_layer, variable
            )
            mask = (
                None
                if constraint is None
                or constraint.mask is None
                or constraint.mask.shape is None
                else constraint.mask.numpy().flatten()
            )
            # print(f"saving variable: {variable.name} {size} {shape}")

            def accumulate_value(dataset, value):
                value = value.flatten()
                if mask is not None:
                    value = numpy.where(mask, value, numpy.nan)

                # dynamically expand dataset as needed
                if dataset.shape[0] < parameter_limit:
                    dataset.resize((parameter_limit, dataset.shape[1]))

                dataset[parameter_index:parameter_limit, sequence_number] = value

            def accumulate_variable(dataset, variable):
                accumulate_value(dataset, variable.numpy())

            accumulate_variable(parameter_dataset, variable)

            for optimizer_member, member_dataset in optimizer_datasets:
                optimizer_variable = get_optimizer_variable(
                    optimizer,
                    optimizer_member,
                    variable,
                )

                if optimizer_variable is None:
                    optimizer_value = numpy.empty(variable.shape)
                    optimizer_value.fill(numpy.nan)
                    accumulate_value(member_dataset, optimizer_value)
                else:
                    accumulate_variable(member_dataset, optimizer_variable)

            parameter_index = parameter_limit

        access_model_parameters.visit_parameters(
            root,
            visit_variable,
        )
    return sequence_number
