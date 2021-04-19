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

    python -u -m dmp.experiment.aspect_test "{'dataset': 'nursery','budgets':[500], 'topologies' : [ 'wide_first' ], 'depths' : [4],  'run_config' : { 'epochs': 2}, 'test_split': 0.1, 'reps': 1 }"

## Extra Eagle Setup Steps

This uses the optimized tensorflow build for Eagle CPUs and GPUs:
        
    pip install --upgrade --force-reinstall /nopt/nrel/apps/wheels/tensorflow-2.3.2-cp38-cp38-linux_x86_64.whl


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
 | python -u -m dmp.jq_enqueue dmp test_tag
```

Note, this allows you to first save the output of aspect_test to a local file for reproducibility purposes.

## Running aspect-test jobs from the job queue

Start a jq_runner.py process pointing at the correct queue, and also 

```
python -u -m dmp.jq_runner dmp test_tag
```


# Experimentson Eagle using Job Queue and Sbatch

## Aspect Test

dmp.experiment.aspect_test

Full Config:
```
{ mode:list, datasets: ['201_pol', '529_pollen', '537_houses', 'adult', 'connect_4', 'mnist', 'nursery', 'sleep', 'wine_quality_white'], 'topologies': ['rectangle', 'trapezoid', 'exponential', 'wide_first'], 'budgets': [1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432], 'depths': [2,3,4,5,7,8,9,10,12,14,16,18,20], 'log':'postgres' }
```

Save smaller test config to a log file:
```
python -u -m dmp.experiment.aspect_test "{ mode:list, datasets: ['201_pol', '529_pollen', '537_houses', 'adult', 'connect_4', 'mnist', 'nursery', 'sleep', 'wine_quality_white'], 'topologies': ['rectangle', 'trapezoid', 'exponential', 'wide_first'], 'budgets': [1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432], 'depths': [2,3,4,5,7,8,9,10,12,14,16,18,20], 'log':'postgres' }" > log/jordan_eagle_1.jobs
```

Push this log file to the queue
```
cat log/jordan_eagle_1.jobs | python -u -m dmp.jq_runner dmp jordan_eagle_1
```

Run use sbatch to run from queue
```
 sbatch sbatchqueuerunner.sh jordan_eagle_1
```

Monitor job progress using the database
 ```
 SELECT * FROM jobqueue WHERE groupname='jordan_eagle_1' AND START_TIME IS NOT NULL;
 ```

 