# DMP
Direct Model Programming

## Install

This sets up the environment:

    conda env create
    conda activate dmp
    conda develop .

If you're using the job queue:

    pip install --editable=git+https://github.nrel.gov/mlunacek/jobqueue-pg@master#egg=jobqueue

Be sure you do not have keras installed in this environment, as it will cause runners to fail due to a namespace conflict when instantiating the DNN optimizer.

## Test

    pytest -s -v tests

## Running experiments locally

    python -u -m dmp.aspect_test "{'datasets': ['nursery'],'budgets':[500], 'topologies' : [ 'wide_first' ], 'depths' : [4],  'run_config' : { 'epochs': 10}, 'test_split': 0.1, 'reps': 1, 'mode':'direct' }"

## Checkpointing

You can add checkpointing and automatic-resuming of model runs by including the "checkpoint_epochs" parameter in the run config. This should be set to an integer number of epochs. The model will be checkpointed after this number of epochs have been completed. By default, the checkpoints will be saved to the directory "./checkpoints". This can be overridden by setting the environment variable $DMP_CHECKPOINT_DIR, which can itself be overridden by the "checkpoint_dir" parmeter in the config.

The name of the checkpoint will be "run_name" if not specified. If run through the job queue, it will be set to the uuid of the job being run. You can name the checkpoint file manually by passing in "jq_uuid" to the configuration.

Note:
- This feature relies on keras-buoy package from PyPi.
- This feature is not compatible with test_split configuration due to the way Keras stores historical losses in callback objects.
- This feature does not restore the random state, so a result from a session which has been checkpointed and resumed may not be reproducible.

    python -u -m dmp.aspect_test "{'datasets': ['nursery'],'budgets':[500], 'topologies' : [ 'wide_first' ], 'depths' : [4],  'run_config' : { 'epochs': 10}, 'reps': 1, 'mode':'direct', 'checkpoint_epochs':1, 'jq_uuid':'nursery_500_widefirst_4' }"

## Tensorboard Logging

You can enable tensorboard logging with 'tensorboard' configuration. Set this to the tensorboard log directory.

```
'tensorboard':'./log/tensorboard'
```

To view the tensorboard logs, use the following command:

```
tensorboard --logdir ./log/tensorboard/
```
