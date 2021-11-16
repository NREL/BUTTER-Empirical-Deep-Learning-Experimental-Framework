# DMP
Direct Model Programming

## Install

This sets up the environment:

    conda env create
    conda activate dmp
    conda develop .

If you're using the job queue:

    pip install --editable=git+https://github.nrel.gov/mlunacek/jobqueue-pg@master#egg=jobqueue

## Test

    pytest -s -v tests

## Running experiments locally

    python -u -m dmp.experiment.aspect_test "{'datasets': ['nursery'],'budgets':[500], 'topologies' : [ 'wide_first' ], 'depths' : [4],  'run_config' : { 'epochs': 10}, 'test_split': 0.1, 'reps': 1, 'mode':'direct' }"

## Checkpointing

You can add checkpointing and automatic-resuming of model runs by including the "checkpoint_epochs" parameter in the run config. This should be set to an integer number of epochs. The model will be checkpointed after this number of epochs have been completed. By default, the checkpoints will be saved to the directory "./checkpoints". This can be overridden by setting the environment variable $DMP_CHECKPOINT_DIR, which can itself be overridden by the "checkpoint_dir" parmeter in the config.

The name of the checkpoint will be "run_name" if not specified. If run through the job queue, it will be set to the uuid of the job being run. You can name the checkpoint file manually by passing in "jq_uuid" to the configuration.

Note:
- This feature relies on keras-buoy package from PyPi.
- This feature is not compatible with test_split configuration due to the way Keras stores historical losses in callback objects.
- This feature does not restore the random state, so a result from a session which has been checkpointed and resumed may not be reproducible.

    python -u -m dmp.experiment.aspect_test "{'datasets': ['nursery'],'budgets':[500], 'topologies' : [ 'wide_first' ], 'depths' : [4],  'run_config' : { 'epochs': 10}, 'reps': 1, 'mode':'direct', 'checkpoint_epochs':1, 'jq_uuid':'nursery_500_widefirst_4' }"

## Tensorboard Logging

You can enable tensorboard logging with 'tensorboard' configuration. Set this to the tensorboard log directory.

```
'tensorboard':'./log/tensorboard'
```

To view the tensorboard logs, use the following command:

```
tensorboard --logdir ./log/tensorboard/
```

## Keras model output (as graphviz / png image)

Note: You must install Graphviz and Pydot to use this feature.

```
'plot_model':'./log/plot'
```

## Residual Networks

DMP supports training of residual networks by using the 'residual_mode' configuration. Set this to "full" for wide_first or rectangular networks only to enable residual mode.

```
'residual_mode':'full'
```

## Extra Eagle Setup Steps

This uses the optimized tensorflow build for Eagle CPUs and GPUs:
    
    pip install --upgrade --force-reinstall /nopt/nrel/apps/wheels/tensorflow-2.4.0-cp38-cp38-linux_x86_64.whl


## Example Eagle Environment Setup Script

An example script that sets up the right modules and environment variables to use the optimized tenseorflow build and appropriate directories. 

    #!/bin/bash
    source ~/.bashrc
    conda deactivate
    
    module purge
    module use /nopt/nrel/apps/modules/centos74/modulefiles/
    module load gcc/7.4.0
    module load cuda/10.0.130
    module load cudnn/7.4.2/cuda-10.0
    module load conda
    
    
    module load conda
    conda deactivate
    
    export TEST_TMPDIR=/scratch/ctripp/tmp
    export TMPDIR=/scratch/ctripp/tmp
    export PYTHONPATH=/projects/dmpapps/ctripp/src
    
    cd /projects/dmpapps/ctripp/src
    conda activate /projects/dmpapps/ctripp/env/dmp


## Running experiments on Eagle with Sbatch

    Create ~/admp file that contains env setup script like above.
    sbatch -n1 -t18:00:00 --gres=gpu:1 ./srundmp.sh dmp.experiment.aspect_test.py "{'dataset': 'nursery','budgets':[262144, 524288, 1048576, 2097152, 4194304, 8388608], 'topologies' : [ 'wide_first' ] }"

# Job Queue

Create the .jobqueue.json file in your home directory

## Enqueue aspect-test jobs using the job queue

Pipe the output of aspect_test.py list into jq_enqueue, with a tag name as the argument

```
python -u -m dmp.experiment.aspect_test "{'mode':'list', 'dataset': 'nursery','budgets':[500, 1000], 'topologies' : [ 'wide_first' ], 'depths' : [4],  'run_config' : { 'epochs': 2}, 'test_split': 0.1, 'reps': 1, 'log':'postgres'}" \
 | python -u -m dmp.jq.jq_enqueue dmp test_tag
```

Note, this allows you to first save the output of aspect_test to a local file for reproducibility purposes.

## Running aspect-test jobs from the job queue

Start a jq_runner.py process pointing at the correct queue, and also 

```
python -u -m dmp.jq_runner dmp test_tag
```


# Experimentson Eagle using Job Queue and Sbatch

## Aspect Test

Python Module: dmp.experiment.aspect_test

Full Config:
```
{ mode:list, datasets: ['201_pol', '529_pollen', '537_houses', 'adult', 'connect_4', 'mnist', 'nursery', 'sleep', 'wine_quality_white'], 'topologies': ['rectangle', 'trapezoid', 'exponential', 'wide_first'], 'budgets': [1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432], 'depths': [2,3,4,5,7,8,9,10,12,14,16,18,20], 'log':'postgres' }
```

Residuals:
```
{ mode:list, datasets: ['201_pol', '529_pollen', '537_houses', 'adult', 'connect_4', 'mnist', 'nursery', 'sleep', 'wine_quality_white'], 'topologies': ['rectangle', 'wide_first'], 'residual_modes': ['full'], 'budgets': [1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432], 'depths': [2,3,4,5,7,8,9,10,12,14,16,18,20], 'log':'postgres' }
```


### Debug / Demo

Save small test config to a log file:
```
python -u -m dmp.experiment.aspect_test "{ mode:list, datasets: ['201_pol', '529_pollen', '537_houses', 'adult', 'connect_4', 'mnist', 'nursery', 'sleep', 'wine_quality_white'], 'topologies': ['rectangle', 'trapezoid', 'exponential', 'wide_first'], 'budgets': [1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432], 'depths': [2,3,4,5,7,8,9,10,12,14,16,18,20], 'log':'postgres' }" > log/jordan_eagle_1.jobs
```

Push this log file to the queue
```
cat log/jordan_eagle_1.jobs | python -u -m dmp.jq.jq_enqueue dmp jordan_eagle_1
```

Queue runner in sbatch
```
 sbatch sbatchqueuerunner.sh dmp jordan_eagle_1
```

Slurm nanny with Python (use Login node as nanny)
```
 screen
 conda activate dmp
 python dmp/jq_slurm.py dmp jordan_eagle_1
```

Monitor job progress using the database
```
SELECT * FROM jobqueue WHERE groupname='jordan_eagle_1' AND START_TIME IS NOT NULL;
```

Monitor SLURM jobs
```
squeue -u username
scancel jobid
```



# Vermillion (LDRD Cluster) Notes

DMP has been successfully run on vermillion.hpc.nrel.gov. These notes should help you get started with Vermillion.

## 1. Activate Anaconda in your shell
```
module use /projects/dmpapps/module/
```

Initialize Anaconda
```
module load conda/mini_py39_4.9.2
conda init
source ~/.bashrc
```

Load cuda module
```
module load cuda
```

## 2. Ensure you have the correct files in your account

Install DMP
```
cd $DMP_HOME
git clone git@github.nrel.gov:ctripp/DMP.git
cd DMP
conda env create
pip install -e .
```

Install Jobqueue
```
cd $DMP_HOME
git clone git@github.nrel.gov:mlunacek/jobqueue-pg.git
cd jobqueue-pg
pip install -e .
```

Copy Job Queue Config File from Eagle into your Vermillion home directory
```
scp eagle:.jobqueue.json ~/.jobqueue.json
```

## 3. Test installation on an interactive node
```
srun --time=30 --account=dmpapps --ntasks=36 --pty $SHELL
python -m dmp.jq.jq_node_manager dmp fixed_3k_1 "[[0,3,0,0,0], [3,6,0,0,0], [6,9,0,0,0], [9,12,0,0,0], [12,15,0,0,0], [15,18,0,0,0], [18,21,0,0,0], [21,24,0,0,0], [24,27,0,0,0], [27,30,0,0,0],[30,33,0,0,0],[33,36,0,0,0]]"
```

## 4. TODO: Modify ADMP and slurm_job_runner.sh to submit batch jobs to Vemrillion.

