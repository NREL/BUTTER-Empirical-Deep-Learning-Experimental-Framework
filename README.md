<img src="star-collage.png?raw=true" alt="Collage image demonstrating several axis of the BUTTER dataset" width=300px />

# The BUTTER Empirical Deep Learning Framework

The BUTTER Empirical Framework enables researchers to run high volumes of computational experiments, including machine learning experiments, in a highly distributed asynchronous way. The BUTTER Framework was designed for asynchronous, unpredictable, and occasionally unreliable worker jobs to execute on any number of computing systems including laptops, servers, cloud resources, clusters, and high-performance supercomputers (HPC systems). 

+ Examples of plotting data from the dataset are available [here](https://github.com/NREL/BUTTER-Better-Understanding-of-Training-Topologies-through-Empirical-Results).

## Using and Citing this Code

**If you benefit from this code or concept, please cite these resources:**

[The paper](https://arxiv.org/abs/2207.12547):

  Tripp, C. E., Perr-Sauer, J., Hayne, L., & Lunacek, M. (2022). An Empirical Deep Dive into Deep Learning's Driving Dynamics. *arXiv preprint arXiv:2207.12547.*

BibTex:

    @article{butter_publication,
      title={An Empirical Deep Dive into Deep Learning's Driving Dynamics},
      author={Tripp, Charles Edison and Perr-Sauer, Jordan and Hayne, Lucas and Lunacek, Monte},
      journal={arXiv preprint arXiv:2207.12547},
      year={2022}
    }

[The dataset](https://dx.doi.org/10.25984/1872441):

  Tripp, Charles, Perr-Sauer, Jordan, Hayne, Lucas, & Lunacek, Monte. *BUTTER - Empirical Deep Learning Dataset*. United States. https://dx.doi.org/10.25984/1872441

    @div{butter_dataset, 
      title = {BUTTER - Empirical Deep Learning Dataset}, 
      author = {Tripp, Charles, Perr-Sauer, Jordan, Hayne, Lucas, and Lunacek, Monte.}, 
      doi = {10.25984/1872441}, 
      url = {https://data.openei.org/submissions/5708},
      place = {United States}, 
      year = {2022}, 
      month = {05}
    }

[The Code](https://doi.org/10.11578/dc.20220608.2)

  Tripp, Charles, Perr-Sauer, Jordan, Lunacek, Monte, & Hayne, Lucas. (2022, May 13). *BUTTER: An Empirical Deep Learning Experimental Framework*. [Computer software]. https://github.com/NREL/BUTTER-Empirical-Deep-Learning-Experimental-Framework. https://doi.org/10.11578/dc.20220608.2.


    @misc{butter_code,
      title = {BUTTER: An Empirical Deep Learning Experimental Framework},
      author = {Tripp, Charles and Perr-Sauer, Jordan and Lunacek, Monte and Hayne, Lucas},
      abstractNote = {BUTTER is a system with which empirical deep learning experiments can be conducted and their results, including training and model performance characteristics, can be accumulated for further analysis.},
      doi = {10.11578/dc.20220608.2},
      url = {https://doi.org/10.11578/dc.20220608.2},
      howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20220608.2}},
      year = {2022},
      month = {may}
    }

The [DOE Code record for JobQueue](https://www.osti.gov/doecode/biblio/74434) which powers BUTTER:

    Tripp, Charles, Perr-Sauer, Jordan, Lunacek, Monte, & Hayne, Lucas. (2022, May 13). *JobQueue: (A Task Queue for Coordinating Varied Tasks Across Multiple HPC Resources and HPC Jobs)*. [Computer software]. https://github.com/NREL/E-Queue-HPC. https://doi.org/10.11578/dc.20220608.1.

        @misc{jobqueue_code,
            title = {JobQueue-PG: A Task Queue for Coordinating Varied Tasks Across Multiple HPC Resources and HPC Jobs},
            author = {Tripp, Charles and Perr-Sauer, Jordan and Lunacek, Monte and Hayne, Lucas},
            abstractNote = {The software allows for queueing and dispatch of tasks of small, varied, or uncertain runtimes across multiple HPC jobs, resources, and other computing systems. The software was designed to allow scientists to enqueue, run, and accumulate results from computational experiments in an efficient, manageable manner. For example, the software can be used to enqueue many small computational experiments and run them using several long-running multi-node HPC jobs that may or may not run simultaneously.},
            doi = {10.11578/dc.20220608.1},
            url = {https://doi.org/10.11578/dc.20220608.1},
            howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20220608.1}},
            year = {2022},
            month = {may}
        }

            

If relevant, also cite our [BUTTER Dataset](https://data.openei.org/submissions/5708) generated using this framework:
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


Next, install JobQueue:

    pip install --editable=git+https://github.com/NREL/E-Queue-HPC.git#egg=jobqueue

Be sure you do not have keras installed in this environment, as it will cause runners to fail due to a namespace conflict when instantiating the DNN optimizer.

For performance reasons, you might consider installing `tensorflow-gpu`, or an optimized `tensorflow` package. If you don't use a GPU-enabled version of TensorFlow, the framework will not be able to use any GPUs.

### Configuring JobQueue

JobQueue will look for a credentials configuration file named `.jobqueue.json` in your home directory as defined by `os.path.join(os.environ['HOME'], ".jobqueue.json")`. The file should look something like this:

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

This [JSON](https://www.json.org/json-en.html) configuration file tells the framework what database to connect to and what credentials to use when doing so. Make sure it matches the database you configured. The `"table_name"` setting sets the prefix of the job status and job data tables. The [database setup script](/database/setup_database.sql) creates tables with prefix `"job"` (`"job_status"` and `"job_data"`). If you wish to use a different prefix, consider editing the file or manually creating the tables. However, upon initialization, JobQueue should create the tables if they do not exist. Once setup, you can verify that the database connection works by running the [database connection test script](dmp/util/check_database.py):


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

To add experiments to the queue, use the `JobQueue.push(job)` method. An example scripts adding an experiment batches are in the [example_batch_scripts](/dmp/example_batch_scripts) directory, such ash [make_depth6_batch](/dmp/example_batch_scripts/make_depth6_batch.py) which enqueues a set of 6 hidden layer neural network training runs over several different sizes, shapes, and datasets.

## Running Workers

A worker process can be executed directly by calling

    python -u -m dmp.jobqueue_interface.worker [first_socket] [num_sockets] [first_core] [num_cores] [first_gpu] [num_gpus] [gpu_mem] [queue]

For example,

    python -u -m dmp.jobqueue_interface.worker 0 1 0 4 0 0 0 1

Will run a worker for queue #1 using the first 4 cores of the first NUMA node (typically corresponding to the first physical CPU socket) in the system.

    python -u -m dmp.jobqueue_interface.worker 0 1 0 1 0 1 8192 1

Will run a worker for queue #1 using the first cores of the first NUMA node in the system and also use 8GB of the first (zeroth) GPU. You can use the [numactl utility](https://linux.die.net/man/8/numactl) to inspect the CPU n## Citing This Dataset
**If you use this dataset, please cite our upcoming dataset publication, which is currently under review for NeurIPS 2022.**

[The paper](https://arxiv.org/abs/2207.12547):

  Tripp, C. E., Perr-Sauer, J., Hayne, L., & Lunacek, M. (2022). An Empirical Deep Dive into Deep Learning's Driving Dynamics. *arXiv preprint arXiv:2207.12547.*

BibTex:

    @article{butter_publication,
      title={An Empirical Deep Dive into Deep Learning's Driving Dynamics},
      author={Tripp, Charles Edison and Perr-Sauer, Jordan and Hayne, Lucas and Lunacek, Monte},
      journal={arXiv preprint arXiv:2207.12547},
      year={2022}
    }

[The dataset](https://dx.doi.org/10.25984/1872441):

  Tripp, Charles, Perr-Sauer, Jordan, Hayne, Lucas, & Lunacek, Monte. *BUTTER - Empirical Deep Learning Dataset*. United States. https://dx.doi.org/10.25984/1872441

    @div{butter_dataset, 
      title = {BUTTER - Empirical Deep Learning Dataset}, 
      author = {Tripp, Charles, Perr-Sauer, Jordan, Hayne, Lucas, and Lunacek, Monte.}, 
      doi = {10.25984/1872441}, 
      url = {https://data.openei.org/submissions/5708},
      place = {United States}, 
      year = {2022}, 
      month = {05}
    }

[The Code](https://doi.org/10.11578/dc.20220608.2)

  Tripp, Charles, Perr-Sauer, Jordan, Lunacek, Monte, & Hayne, Lucas. (2022, May 13). *BUTTER: An Empirical Deep Learning Experimental Framework*. [Computer software]. https://github.com/NREL/BUTTER-Empirical-Deep-Learning-Experimental-Framework. https://doi.org/10.11578/dc.20220608.2.


    @misc{butter_code,
      title = {BUTTER: An Empirical Deep Learning Experimental Framework},
      author = {Tripp, Charles and Perr-Sauer, Jordan and Lunacek, Monte and Hayne, Lucas},
      abstractNote = {BUTTER is a system with which empirical deep learning experiments can be conducted and their results, including training and model performance characteristics, can be accumulated for further analysis.},
      doi = {10.11578/dc.20220608.2},
      url = {https://doi.org/10.11578/dc.20220608.2},
      howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20220608.2}},
      year = {2022},
      month = {may}
    }
The framework also has a [node manager script](dmp/jq/jq_node_manager.py) that will allocate workers over the hardware in a reasonable manner for many deep learning experiments. In order to use the node manager, the system must be linux-like, supporting bash scripts and the numactl utility must be installed and working.

One worker will be allocated for up to every 64 cores of each NUMA node, and two cores from the same NUMA node will be allocated for each GPU worker in the system, evenly distributed over the NUMA nodes. A GPU worker will be allocated for approximately every 6.5GB of GPU memory available, up to 4 workers per GPU. These parameters can be changed by editing the script.

The node manager is convenient for use in HPC job systems like slurm where resources are allocated on a per-node basis. In this case, submitting jobs that run the node manager on each node will cause multiple workers to be automatically allocated over the node's resources. An example slurm script is [slurm_job_runner.sh](slurm_job_runner.sh). Additionally, the node manager will call a [worker manager](dmp/jobqueue_interface/worker_manager.py) script that acts as a nanny, restarting the worker process if it exits abnormally and therefore avoiding spurious worker failures. Finally, GPU and CPU based jobs can be made to use different conda environments or generally to execute under different parameters by supplying a `custom_cpu_run_script.sh` and/or a `custom_gpu_run_script.sh` which will be executed before running the `worker.py` script. Custom scripts can be adapted from the default [cpu run script](cpu_run_script.sh).



