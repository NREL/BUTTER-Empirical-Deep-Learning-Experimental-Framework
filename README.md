# DMP
Direct Model Programming

## Install

This sets up the environment:

        conda env create
        conda activate dmp
        conda develop .

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
    export PYTHONPATH=/scratch/ctripp/src/dmp
    
    cd /projects/dmpapps/ctripp/src
    conda activate /projects/dmpapps/ctripp/env/dmp

## Test

        pytest -s -v tests

        