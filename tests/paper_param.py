import numpy as np

LMCRN20 = {
        # https://github.com/facebookresearch/open_lth/blob/main/models/cifar_resnet.py
        "Standard": {
                 "data": "CIFAR10",
                 "train_Step": 63e3,
                 "batch_size": 128,
                 "momentum": .1,
                 "learning_rate": {
                        "class": "PiecewiseConstantDecay", 
                        "boundaries": [32e3, 48e3],
                        "values": [.1, .01, .001]
                 },
                 "warmup": 0,
                 "prune_Density": .168,
                 "optimizer": "SGD"
        },

        "Low": {
            "data": "CIFAR10",
            "train_Step": 63e3,
            "batch_size": 128,
            "momentum": .01,
            "learning_rate": {
                "class": "PiecewiseConstantDecay",
                "boundaries": [32e3, 48e3],
                "values": [.01, .001, .0001]
            },
            "warmup": 0,
            "prune_Density": .086,
            "optimizer": "SGD"
        },

        "Warmup": {
            "data": "CIFAR10",
            "train_Step": 63e3,
            "batch_size": 128,
            "momentum": .03,
            "learning_rate": {
                "class": "PiecewiseConstantDecay",
                "boundaries": np.concatenate( (np.linspace(1, 32e3, num=int(32e3-1)), np.array( [32e3, 48e3]       )  )),
                "values":     np.concatenate( ( np.linspace(0,   .03, num=int(32e3))  , np.array( [.03  , .003, .0003] )  ))
            },
            "warmup": 30e3,
            "prune_Density": .086,
            "optimizer": "SGD"
        }                
    }

LMCVGG = {
        # https://github.com/facebookresearch/open_lth/blob/main/models/cifar_vgg.py
        "Standard": {
                 "data": "CIFAR10",
                 "train_Step": 63e3,
                 "batch_size": 128,
                 "momentum": .1,
                 "learning_rate": {
                        "class": "PiecewiseConstantDecay", 
                        "boundaries": [32e3, 48e3],
                        "values": [.1, .01, .001]
                 },
                 "warmup": 0,
                 "prune_Density": .015,
                 "optimizer": "SGD"
        },
        
        "Low": {
                 "data": "CIFAR10",
                 "train_Step": 63e3,
                 "batch_size": 128,
                 "momentum": .01,
                 "learning_rate": {
                        "class": "PiecewiseConstantDecay", 
                        "boundaries": [32e3, 48e3],
                        "values": [.01, .001, .0001]
                 },
                 "warmup": 0,
                 "prune_Density": .055,
                 "optimizer": "SGD"
        },
        
        "Warmup": {
                 "data": "CIFAR10",
                 "train_Step": 63e3,
                 "batch_size": 128,
                 "momentum": .1,
                 "learning_rate": {
                        "class": "PiecewiseConstantDecay", 
                        "boundaries": np.concatenate( (np.linspace(1, 32e3, num=int(32e3)), np.array( [32e3, 48e3]       )  )),
                        "values":     np.concatenate( (np.linspace(0,   .03, num=int(32e3))  , np.array( [.1  , .01, .001] )  ))
                 },
                 "warmup": 30e3,
                 "prune_Density": .015,
                 "optimizer": "SGD"
        },
}


PaperParams = { 
    # https://github.com/facebookresearch/open_lth
    "Linear_Mode_Connectivity": {"RESNET": LMCRN20, "VGG16": LMCVGG} 
}

def get_paper_param(paper, model, param):
    return PaperParams[paper][model][param]