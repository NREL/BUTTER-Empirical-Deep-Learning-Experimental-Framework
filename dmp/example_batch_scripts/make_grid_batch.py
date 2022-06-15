"""
Enqueues jobs from stdin into the JobQueue
"""

import argparse

from command_line_tools import command_line_config
from dmp.jobqueue_interface.common import jobqueue_marshal

import sys

from dmp.task.aspect_test.aspect_test_task import AspectTestTask


def do_parameter_sweep(sweep_config, task_handler):

    repetitions = sweep_config['repetitions']
    sweep_values = sweep_config['sweep_values']

    keys = list(sweep_values.keys())
    task_config = {}
    priority = sweep_config['base_priority']

    def do_sweep(key_index):
        nonlocal priority, task_config, keys
        if key_index < 0:
            for rep in range(repetitions):
                task_config['seed'] = priority
                task = AspectTestTask(**task_config)
                task_handler(task)
                priority += 1
        else:
            key = keys[key_index]
            for v in sweep_values[key]:
                task_config[key] = v
                do_sweep(key_index-1)

    do_sweep(len(keys)-1)


def main():
    default_config = {
        'repetitions': 1,
        'base_priority': 0,
        'sweep_values': {
            'batch': ['test'],
            'dataset': ['529_pollen'],
            'input_activation': ['relu'],
            'activation': ['relu'],
            'optimizer': [{'class_name': 'adam', 'learning_rate': 0.0001}],
            'shape': ['rectangle'],
            'size': [1048576],
            'depth': [3],
            'validation_split': [.2],
            'validation_split_method': ['shuffled_train_test_split'],
            'run_config': [{
                'shuffle': True,
                'epochs': 3000,
                'batch_size': 256,
                'verbose': 0,
            }],
            'label_noise': [0.0],
            'kernel_regularizer': [None],
            'bias_regularizer': [None],
            'activity_regularizer': [None],
            'early_stopping': [None],
            'save_every_epochs': [None],
        },
    }

    sweep_config = command_line_config.parse_config_from_args(
        sys.argv[1:], default_config)

    def handler(task):
        marshaled_task = jobqueue_marshal.marshal(task)
        print(marshaled_task)

    do_parameter_sweep(sweep_config, handler)


if __name__ == "__main__":
    main()
