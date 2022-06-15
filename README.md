# The BUTTER Empirical Deep Learning Framework

The BUTTER Empirical Framework enables researchers to run high volumes of computational experiments, including machine learning experiments, in a highly distributed asynchronous way. The BUTTER Framework was designed for asynchronous, unpredictable, and occasionally unreliable worker jobs to execute on any number of computing systems including laptops, servers, cloud resources, clusters, and high-performance supercomputers (HPC systems). 

## Using and Citing this Code

**If you benefit from this code or concept, please cite these resources:**

+ Our upcoming dataset publication, which is currently under review for the [NeurIPS 2022 Datasets and Benchmarks Track](https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks).

+ The Framework's [DOE Code record](https://www.osti.gov/doecode/biblio/74457): 

    Tripp, Charles, Perr-Sauer, Jordan, Lunacek, Monte, & Hayne, Lucas. (2022, May 13). *BUTTER Empirical Deep Learning Experimental Framework.* [Computer software]. https://github.com/NREL/BUTTER-Empirical-Deep-Learning-Experimental-Framework.  https://doi.org/10.11578/dc.20220608.2.

+ The [DOE Code record for JobQueue-PG](https://www.osti.gov/doecode/biblio/74434) which powers BUTTER:
    Tripp, Charles, Perr-Sauer, Jordan, Lunacek, Monte, & Hayne, Lucas. (2022, May 13). *JobQueue-PG: (A Task Queue for Coordinating Varied Tasks Across Multiple HPC Resources and HPC Jobs)*. [Computer software]. https://github.com/NREL/E-Queue-HPC. https://doi.org/10.11578/dc.20220608.1.

+ If relevant, also cite our [BUTTER Dataset](https://data.openei.org/submissions/5708) generated using this framework:
    Tripp, Charles, Perr-Sauer, Jordan, Hayne, Lucas, & Lunacek, Monte. *BUTTER - Empirical Deep Learning Dataset.* United States. https://data.openei.org/submissions/5708


## The BUTTER Empirical Deep Learning Dataset

Utilizing the BUTTER Framework, we generated the [BUTTER Empirical Deep Learning Dataset](https://data.openei.org/submissions/5708), an empirical dataset of the deep learning phenomena on dense fully connected networks, scanning across thirteen datasets, eight network shapes, fourteen depths, twenty-three network sizes (number of trainable parameters), four learning rates, six minibatch sizes, four levels of label noise, and fourteen levels of L1 and L2 regularization each. In total, this dataset covers 178 thousand distinct hyperparameter settings ("experiments"), 3.55 million individual training runs (an average of 20 repetitions of each experiments), and a total of 13.3 billion training epochs (three thousand epochs were covered by most runs). Accumulating this dataset consumed 5,448.4 CPU core-years, 17.8 GPU-years, and 111.2 node-years.

## Setup and Installation


### Database Setup
Before running the framework, install a PostgreSQL database using [the instructions in the postgresql documentation](https://www.postgresql.org/docs/current/). This database will act as the central distribution and collection point for experimental runs. Runs are enqueued into the database, distributed to worker processes which execute the runs, and the results of each run is recorded back into this database.

Once operational, [create a "database" within the PostgreSQL instance](https://www.postgresql.org/docs/current/sql-createdatabase.html).


    CREATE DATABASE project_1;
    

Once created, connect to the database with your favorite PostgreSQL tool and execute the [database setup script](/database/setup_database.sql). This script will create the appropriate tables and indexes for the framework to work properly.

### Conda Environment Creation

To run the framework code, setup a [Conda](https://docs.conda.io/en/latest/) environment using the [environment.yml](environment.yml) configuration file:

    conda env create --name butter --file environment.yml
    conda activate butter
    conda develop .

If Conda is too slow you might have better luck using [Mamba](https://github.com/mamba-org/mamba) as a drop-in replacement. The "develop" command adds the repository root to the python module path for the environment. If doing so is troublesome, you can also use [pip](https://pip.pypa.io/en/stable/) to do (almost) the same thing:


    pip install -e . 


Next, install JobQueue-PG:

    pip install --editable=git+https://github.com/NREL/E-Queue-HPC.git#egg=jobqueue

Be sure you do not have keras installed in this environment, as it will cause runners to fail due to a namespace conflict when instantiating the DNN optimizer.

### Configuring JobQueue-PG

JobQueue-PG will look for a credentials configuration file named `.jobqueue.json` in your home directory as defined by `os.path.join(os.environ['HOME'], ".jobqueue.json")`. The file should look something like this:

    {
        "project_1": {
            "host": "my.database.nrel.gov",
            "user": "project_1_username",
            "database": "project_1_database",
            "password": "project_1_password",
            "table_name" : "job"
        }
        "project_2": {
            "host": "my.database.nrel.gov",
            "user": "project_2_username",
            "database": "project_2_database",
            "password": "project_2_password",
            "table_name" : "job"
        },
    }

This [JSON](https://www.json.org/json-en.html) configuration file tells the framework what database to connect to and what credentials to use when doing so. Make sure it matches the database you configured. The `"table_name"` setting sets the prefix of the job status and job data tables. The [database setup script](/database/setup_database.sql) creates tables with prefix `"job"` (`"job_status"` and `"job_data"`). If you wish to use a different prefix, consider editing the file or manually creating the tables. However, upon initialization, JobQueue-PG should create the tables if they do not exist. Once setup, you can verify that the database connection works by running the [database connection test script](dmp/util/check_database.py):


    python -m dmp.util.check_database project_1

If all is well, you will see an output like:

    checking database connection for project "project_1"...
    {'host': 'my.database.nrel.gov', 'user': 'project_1_username', 'database': 'project_1_database', 'password': 'project_1_password'}
    Credentials loaded, attempting to connect.
    All is well.
    Done.

If jobs are already enqueued, this script will additionally list a summary of queued jobs.

## Testing 

You can verify the basic operation of the framework with pytest:

    pytest -s -v tests

## Enqueueing Experimental Runs


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
