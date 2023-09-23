import numpy as np

LMCRN20 = {
    # https://github.com/facebookresearch/open_lth/blob/main/models/cifar_resnet.py
    "standard": {
        "data": "CIFAR10",
        "train_step": 63e3,
        "batch": 128,
        "momentum": 0.1,
        "learning_rate": {
            "class": "PiecewiseConstantDecay",
            "boundaries": [32e3, 48e3],
            "values": [0.1, 0.01, 0.001],
        },
        "warmup": 0,
        "pruning_rate": 0.168,
        "optimizer": "SGD",
    },
    "low": {
        "data": "CIFAR10",
        "train_step": 63e3,
        "batch": 128,
        "momentum": 0.01,
        "learning_rate": {
            "class": "PiecewiseConstantDecay",
            "boundaries": [32e3, 48e3],
            "values": [0.01, 0.001, 0.0001],
        },
        "warmup": 0,
        "pruning_rate": 0.086,
        "optimizer": "SGD",
    },
    "warmup": {
        "data": "CIFAR10",
        "train_step": 63e3,
        "batch": 128,
        "momentum": 0.03,
        "learning_rate": {
            # https://github.com/facebookresearch/open_lth/blob/2ce732fe48abd5a80c10a153c45d397b048e980c/training/optimizers.py#L48
            "class": "PiecewiseConstantDecay",
            "boundaries": list(
                np.concatenate(
                    (np.linspace(1, 32e3, num=int(32e3 - 1)), np.array([32e3, 48e3]))
                )
            ),
            "values": list(
                np.concatenate(
                    (
                        np.linspace(0, 0.03, num=int(32e3 - 1)),
                        np.array([0.03, 0.003, 0.0003]),
                    )
                )
            ),
        },
        "warmup": 30e3,
        "pruning_rate": 0.086,
        "optimizer": "SGD",
    },
}

LMCVGG = {
    # https://github.com/facebookresearch/open_lth/blob/main/models/cifar_vgg.py
    "standard": {
        "data": "CIFAR10",
        "train_step": 63e3,
        "batch": 128,
        "momentum": 0.1,
        "learning_rate": {
            "class": "PiecewiseConstantDecay",
            "boundaries": [32e3, 48e3],
            "values": [0.1, 0.01, 0.001],
        },
        "warmup": 0,
        "pruning_rate": 0.015,
        "optimizer": "SGD",
    },
    "low": {
        "data": "CIFAR10",
        "train_step": 63e3,
        "batch": 128,
        "momentum": 0.01,
        "learning_rate": {
            "class": "PiecewiseConstantDecay",
            "boundaries": [32e3, 48e3],
            "values": [0.01, 0.001, 0.0001],
        },
        "warmup": 0,
        "pruning_rate": 0.055,
        "optimizer": "SGD",
    },
    "warmup": {
        "data": "CIFAR10",
        "train_step": 63e3,
        "batch": 128,
        "momentum": 0.1,
        "learning_rate": {
            "class": "PiecewiseConstantDecay",
            # https://github.com/facebookresearch/open_lth/blob/2ce732fe48abd5a80c10a153c45d397b048e980c/training/optimizers.py#L48
            "boundaries": np.concatenate(
                (np.linspace(1, 32e3, num=int(32e3 - 1)), np.array([32e3, 48e3]))
            ).tolist(),
            "values": np.concatenate(
                (np.linspace(0, 0.03, num=int(32e3 - 1)), np.array([0.1, 0.01, 0.001]))
            ).tolist(),
        },
        "warmup": 30e3,
        "pruning_rate": 0.015,
        "optimizer": "SGD",
    },
}


PaperParams = {
    # https://github.com/facebookresearch/open_lth
    "Linear_Mode_Connectivity": {"RESNET": LMCRN20, "VGG16": LMCVGG}
}


def get_paper_param(paper, model, param):
    return PaperParams[paper][model][param]
