import dataclasses
import os
import subprocess
import sys
from typing import List

import h5py as h5
import hdf5plugin

import dmp.keras_interface.model_serialization as model_serialization
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch


def split_monotonically_increasing(tuples):
    if not tuples:
        return []

    current_sublists = None
    result = []

    for i, current_tuple in enumerate(tuples):
        prev_src, prev_dst = tuples[i - 1]
        current_src, current_dst = tuples[i]

        if i == 0 or any(
            (
                previous_value >= current_value
                for previous_value, current_value in zip(tuples[i - 1], current_tuple)
            )
        ):
            current_sublists = ([], [])
            result.append(current_sublists)

        for sublist, value in zip(current_sublists, current_tuple):  # type: ignore
            sublist.append(value)

    return result


def recompress_model_data_file(filename):
    try:
        src_path = os.path.join(model_serialization.model_data_dir, filename)
        dst_path = os.path.join(model_serialization.model_data_dir, filename + ".dst")
        print(f"{filename}: loading...")
        with h5.File(src_path, "r") as src_file:
            (
                src_epoch_dataset,
                src_parameter_dataset,
                src_optimizer_datasets,
            ) = model_serialization.get_datasets_from_model_file(src_file, None)

            src_epochs = model_serialization.convert_epoch_dataset_into_epochs(
                src_epoch_dataset
            )
            # print(src_epochs)
            # print(f"\n\n")

            retained_map = {}
            for epoch in src_epochs:
                if not ((epoch.marker != 0) or (epoch.fit_number == 0)):
                    continue
                e = dataclasses.replace(epoch)
                e.sequence_number = None
                if e in retained_map:
                    if retained_map[e].sequence_number >= epoch.sequence_number:
                        continue
                retained_map[e] = epoch

            retained_epochs: List[TrainingEpoch] = sorted(retained_map.values())
            num_retained_epochs = len(retained_epochs)

            if num_retained_epochs == len(src_epochs):
                print(f"{filename} already minimized.")
                return False

            # print(retained_epochs)

            # for epoch in retained_epochs:
            #     print(epoch)

            # print(
            #     f"Num retained epochs: {num_retained_epochs} Ratio: {num_retained_epochs / len(src_epochs)}"
            # )

            with h5.File(dst_path, "w") as dst_file:
                (
                    dst_epoch_dataset,
                    dst_parameter_dataset,
                    dst_optimizer_datasets,
                ) = model_serialization.get_datasets_from_model_file(
                    dst_file, [member for member, dataset in src_optimizer_datasets]
                )

                dataset_mapping = [
                    ("epoch_dataset", src_epoch_dataset, dst_epoch_dataset),
                    ("parameter_dataset", src_parameter_dataset, dst_parameter_dataset),
                ] + [
                    (src_member, src_dataset, dst_dataset)
                    for (src_member, src_dataset), (dst_member, dst_dataset) in zip(
                        src_optimizer_datasets, dst_optimizer_datasets
                    )
                ]

                for dataset_name, src_dataset, dst_dataset in dataset_mapping:
                    # print(
                    #     f"resize {dataset_name} from {src_dataset.shape} to ({src_dataset.shape[0], num_retained_epochs})"
                    # )
                    dst_dataset.resize((src_dataset.shape[0], num_retained_epochs))

                index_mapping = [
                    (epoch.sequence_number, dst_sequence_number)
                    for dst_sequence_number, epoch in enumerate(retained_epochs)
                ]

                monotonic_mappings = split_monotonically_increasing(index_mapping)
                for datset_name, src_dataset, dst_dataset in dataset_mapping:
                    print(f"{filename}: copying {datset_name}...")
                    for src_indexes, dst_indexes in monotonic_mappings:
                        dst_dataset[:, dst_indexes] = src_dataset[:, src_indexes]

                # for dst_sequence_number, epoch in enumerate(retained_epochs):
                #     src_sequence_number = epoch.sequence_number
                #     # print(f"copying {src_sequence_number} to {dst_sequence_number}.")
                #     for datset_name, src_dataset, dst_dataset in dataset_mapping:
                #         # print(f"copying {datset_name}...")
                #         dst_dataset[:, dst_sequence_number] = src_dataset[
                #             :, src_sequence_number
                #         ]
                #     dst_epoch_dataset[3, dst_sequence_number] = epoch.marker

            print(f"{filename}: done writing. Renaming...")

            delete_path = src_path + ".del"
            subprocess.run(["mv", src_path, delete_path])
            subprocess.run(["mv", dst_path, src_path])
            subprocess.run(["rm", delete_path])

            print(f"{filename}: Done.")
            return True
    except OSError as error:
        print(f"{filename}: Caught {error}")
        return False
    except e:
        print(f"{filename}: Caught {e}")
        import traceback

        print(traceback.format_exc())
        return False


def recompress_model_data():
    lines = []
    with open(sys.argv[1]) as file:
        lines = [f"{line.rstrip()}.h5" for line in file]
        print(f"loaded {len(lines)} files to recompress...")

    import multiprocessing

    with multiprocessing.Pool(int(sys.argv[2])) as pool:
        results = pool.imap_unordered(recompress_model_data_file, lines)
        for r in results:
            pass

    print("Done!")
    # recompress_model_data_file(sys.argv[1])
    # recompress_model_data_file("ffd1a61a-c567-4978-b912-2f53cec3d77f.h5")


if __name__ == "__main__":
    recompress_model_data()
