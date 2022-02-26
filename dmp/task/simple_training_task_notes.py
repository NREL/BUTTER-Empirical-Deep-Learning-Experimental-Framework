# from abc import abstractmethod

# from attr import dataclass

# @dataclass
# class TrainingTask:
#     # all common settings for training tasks
#     pass

#     @abstractmethod
#     def run(self):
#         # does the thing
#         pass

# @dataclass
# class SimpleTrainingTask(TrainingTask):
#     seed: int
#     log: str
#     dataset: str
#     activation: str

#     # ...

#     def run(self):
#         # does the thing
#         pass

# task = SimpleTrainingTask(123, 'asdf', 'dataset', 'relu')
# task_json = marshal.marshal(task)
# # queue in database
# task_json = {
#     '%': 'SimpleTrainingTask',
#     'seed': 123,
#     'log' : 'asdf',
#     'dataset': 'dataset',
#     'activation': 'relu',
#     'my_set' : {
#         '%':'tuple',
#         ':': [1, 2, 3, 4]
#     }
# }







# @dataclass
# class LayerGrowthTrainingTask(TrainingTask):
#     start_layers: int
#     growth_policy: GrowthHandler

#     def run(self):
#         # does the thing
#         pass


# @dataclass
# class GrowthHandler:

#     @abstractmethod
#     def grow(self, network):
#         pass


# @dataclass
# class AddOnOutputLayerGrowthHandler(GrowthHandler):
#     pass

#     def grow(self, network):
#         # add a layer at the output side...
#         pass

# @dataclass
# class DMPTrainIteration:

#     def run(self):
#         # read previous results (if any)
#         # update generative model
#         # generate new candidates (new structures and parameters)
#         #   - training epochs, or hybridize or mutate structures
#         #   - enqueue candidate training & test runs
#         pass

# @dataclass
# class DMPCandidateTask:

#     def run(self):
#         # initialize uninitialized parameters
#         # apply any perturbations
#         # train network a bit
#         # -- split tasks?
#         # test network
#         # record result
#         # if this is the last task, enqueue next DMPTrainIteration
#         pass


# """

# {
#     "%" : "SimpleTrainingTask",
#     "seed": 1234,
#     "log" : "./log",
#     "dataset": "mnist",
#     "settings" : { "asdf":1234 },
# }

# SimpleTrainingTask

# """
