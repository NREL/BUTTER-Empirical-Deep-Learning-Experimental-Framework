import os
from typing import Dict, List, Optional, Sequence, Sequence, Tuple
from uuid import UUID

import numpy
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch

import h5py as h5
import hdf5plugin

model_data_dir = os.path.join(os.getcwd(), "model_data")

max_epochs = int(1e9)
max_parameters = int(100e9)

os.makedirs(model_data_dir, exist_ok=True)

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


def get_model_data_path(
    run_id: UUID,
) -> str:
    return os.path.join(model_data_dir, f"{str(run_id)}.h5")


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
        chunks=(512, 32),
        maxshape=(max_parameters, max_epochs),
        fillvalue=numpy.nan,
        **hdf5plugin.Blosc(cname="lz4", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE),
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
    if optimizer_members is None:
        optimizer_members = [member for member in optimizer_members_group.keys()]
    optimizer_datasets = [
        (
            member,
            require_parameter_dataset(optimizer_members_group, member, numpy.float16),
        )
        for member in optimizer_members
    ]

    return (
        epoch_dataset,
        parameter_dataset,
        optimizer_datasets,  # type: ignore
    )


def convert_epoch_dataset_into_epochs(epoch_dataset):
    global_epoch = epoch_dataset[0, :]
    fit_number = epoch_dataset[1, :]
    fit_epoch = epoch_dataset[2, :]
    epoch_marker = epoch_dataset[3, :]

    epochs = []
    for i in range(epoch_dataset.shape[1]):
        epochs.append(
            TrainingEpoch(
                global_epoch[i], fit_number[i], fit_epoch[i], epoch_marker[i], i
            )
        )
    epochs.sort()
    return epochs
