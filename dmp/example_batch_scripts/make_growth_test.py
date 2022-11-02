"""
Enqueues jobs from stdin into the JobQueue
"""

import argparse
import time

import jobqueue.connect as connect
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue
import numpy

from command_line_tools import command_line_config
from dmp.jobqueue_interface.common import jobqueue_marshal

import sys

from dmp.task.growth_test.growth_test_task import GrowthTestTask
from dmp.task.aspect_test.aspect_test_utils import find_best_layout_for_budget_and_depth


def do_parameter_sweep(sweep_config, task_handler):

    repetitions = sweep_config['repetitions']
    sweep_values = sweep_config['sweep_values']

    keys = list(sweep_values.keys())
    task_config = {}
    seed = int(time.time())

    def do_sweep(key_index):
        nonlocal task_config, keys, seed
        if key_index < 0:
            for rep in range(repetitions):
                task_config['seed'] = seed
                task = GrowthTestTask(**task_config)
                task_handler(task)
                seed += 1
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
        'queue': -2,
        'sweep_values': {
            'batch': ['growth_test_1'],
            'dataset': ['201_pol'], # , '529_pollen', '537_houses', 'adult', 'connect_4', 'mnist', 'sleep', 'wine_quality_white',
            'input_activation': ['relu'],
            'activation': ['relu'],
            'optimizer': [{'class_name': 'adam', 'config': {'learning_rate': 0.001}}],
            'shape': ['rectangle'], # , 'exponential', 'trapezoid', 'rectangle_residual'
            'size': [256],
            'depth': [3],
            'test_split': [.2],
            'val_split': [.1],
            'test_split_method': ['shuffled_train_val_test_split'],
            'run_config': [{
                'shuffle': True,
                'epochs': 100,
                'batch_size': 256,
                'verbose': 0,
            }],
            'label_noise': [0.0],
            'kernel_regularizer': [None],
            'bias_regularizer': [None],
            'activity_regularizer': [None],
            'early_stopping': [None],
            'save_every_epochs': [None],
            'growth_trigger': ['EarlyStopping'],
            'growth_trigger_params': [{'patience':10}],
            'growth_method': ['grow_network'],
            'growth_method_params': [None],
            'growth_scale': [2.0],
            'max_size': [2048],
        },
    }

    sweep_config = command_line_config.parse_config_from_args(
        sys.argv[1:], default_config)

    tasks = []

    def handler(task):
        nonlocal tasks
        tasks.append(task)

    do_parameter_sweep(sweep_config, handler)

    tasks = sorted(tasks, key=lambda t: (
        t.shape, t.depth, t.dataset, t.size, t.seed))

    base_priority = sweep_config['base_priority']
    jobs = [Job(
        priority=base_priority+i,
        command=jobqueue_marshal.marshal(t),
    ) for i, t in enumerate(tasks)]

    print(f'Generated {len(jobs)} jobs.')
    credentials = connect.load_credentials('dmp')
    queue_id = int(sweep_config['queue'])
    job_queue = JobQueue(credentials, queue_id, check_table=False)
    job_queue.push(jobs)
    print(f'Enqueued {len(jobs)} jobs.')


if __name__ == "__main__":
    main()
