import numpy
import pandas
import io
import uuid
import psycopg.sql
import pyarrow
import pyarrow.parquet
import os
import json
import paramiko
from tqdm import tqdm
# import deepcopy
import copy
import jobqueue
from jobqueue.connection_manager import ConnectionManager
import warnings
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from methods import *
import dmp.keras_interface.model_serialization as model_serialization
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch

pd.options.display.max_seq_items = None

credentials = jobqueue.load_credentials("dmp")

from dataclasses import dataclass

@dataclass
class Column:
    name : str
    pandas_type : str



def getWeights(runId, userName, weightName=None, saveDir="./tempWeights"):
    # Define the location of the weights on the remote server
    location = f"/projects/modularai/dmp/dmp/model_data/{runId}.h5"
    # if the save directory doesn't exist, create it
    os.makedirs(saveDir, exist_ok=True)

    # If weightName is not provided, use runId as weightName
    if weightName is None:
        weightName = runId

    # Define the list of HPC servers
    hpclist = ["kestrel.hpc.nrel.gov"]
        # It is all migrated to kestrel, "vermilion.hpc.nrel.gov", "eagle.hpc.nrel.gov"]

    # Iterate through each HPC server
    for hpc in hpclist:
        command = f"scp {userName}@{hpc}:{location} {os.path.join(saveDir, f'{weightName}.h5')}"
        print(command)
        os.system(command)
    
    return os.path.join(saveDir, f'{weightName}.h5')

def flatten_json(json_obj, parent_key="", separator="_"):
    flattened = {}
    for key, value in json_obj.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key, separator=separator))
        else:
            flattened[new_key] = value
    return flattened

def sort_epochs_weights(epoch_dataset, history):
    epochs = []
    global_epoch = epoch_dataset[0, :]
    fit_number = epoch_dataset[1, :]
    fit_epoch = epoch_dataset[2, :]
    epoch_marker = epoch_dataset[3, :]
    for i in range(epoch_dataset.shape[1]):
        # TrainingEpoch is an object that contains the attributes 
        epoch = TrainingEpoch(
            global_epoch[i], fit_number[i], fit_epoch[i], epoch_marker[i]
        )
        retained =  history
        if retained.loc[
            (retained["epoch"] == epoch.epoch)
            & (retained["fit_number"] == epoch.fit_number)
            & (retained["fit_epoch"] == epoch.fit_epoch)
        ].empty:
            continue
        epochs.append(
            (
                epoch,
                i,
            )
        )
    epochs.sort()
    return epochs

def loadWeights(fileName:str, history):
    import h5py as h5
    print("Loading weights")
    file = h5.File(fileName, 'r')
    epoch_dataset, parameter_dataset, optimizer_datasets = model_serialization.get_datasets_from_model_file(file, None)

    print("sorting weights")
    epochs = sort_epochs_weights(epoch_dataset, history)
    # raise warning the  file needs to be closed
    warnings.warn("File needs to be closed")
    return epochs, parameter_dataset, optimizer_datasets, file
        
# ========
def runQuery(query:str):
    """
    
        Given a query, run the query and return the results as a pandas dataframe
        input:
            query: str
                The query to run
    """
    experiment_id = 0
    run_command = {}
    dataframes = []
    
    os.makedirs("data_artifacts", exist_ok=True)
    with ConnectionManager(credentials) as connection:
        with connection.cursor(binary=True) as cursor:
            cursor.execute(query, binary=True)

            for row in cursor:
                row_data = []
                for column in row:
                    row_data.append(column)
                dataframes.append(pd.DataFrame(row_data).T)      
                
    history = pd.concat(dataframes, ignore_index=True, axis=0)
    return history
    
def getParentRuns(data:str = "lmc_mnist_lenet_4"):
    """
        Given an experiment, get the parent runs that are associated with the experiment
        input:
            data: str
                The name of the experiment
        output: 
            parentRuns: pd.DataFrame
                The dataframe containing the parent runs
    """
    query = f"""SELECT
                id
            FROM
                run parent
            WHERE TRUE
                AND command @@ '$."experiment"."type" == "LTHExperiment"' -- using 'jsonpath' expression to filter
                AND command @@ '$."config"."data"."batch" == "lmc_mnist_lenet_4"' -- using 'jsonpath' expression to filter
                AND status >= 2
                AND queue >= 0
            """
    fileName = f"./data_artifacts/parent_runs_{data}.csv"
    if not os.path.exists(fileName):
        parentRuns = runQuery(query)
        parentRuns.columns = ['id']
        
    if os.path.exists(fileName):
        parentRuns = pd.read_csv(fileName)
        
    return parentRuns

def getExperiments(parentID:str, reDownload=False):
    """
        Given a parentID, get the experiments that are associated with the parentID
        input: 
            parentID: str
                The parentID to use to get the experiments

        output:
            experiments: pd.DataFrame
                The dataframe containing the experiments
    """
    query="""
            SELECT
            id,
            command #> '{experiment,pruning,method,type}' pruning_method, 
            command #> '{experiment,pruning,method,pruning_rate}' pruning_rate, 
            command #> '{experiment,pruning,rewind_epoch,epoch}' rewind_epoch, 
            command #> '{config,seed}' random_seed, 
            experiment_id,
            command,
            history
        FROM
            run r
        WHERE TRUE
            AND command @> '{"experiment":{"type":"IterativePruningExperiment"}}'
            AND (command #> '{experiment,pruning,method,pruning_rate}')::float >= 0.1
            AND status >= 2
            AND queue >= 0
        """ + f"AND parent_id = '{parentID}';"
    fileName = f"./data_artifacts/experiments_{parentID}.csv"
    if os.path.exists(fileName):
        experiments = pd.read_csv(fileName)
        experiments['history'] = experiments['history'].apply(lambda x: pd.read_json(x))
        
    if reDownload or not os.path.exists(fileName):
        experiments = runQuery(query)
        experiments.columns = ['id', 'pruning_method', 'pruning_rate', 'rewind_epoch', 'random_seed', 'experiment_id', 'command', 'history']

        for row in experiments.iterrows():
            run_command = row[1]['command']
            from dmp.marshaling import marshal
            run = marshal.demarshal(run_command)
            flat = flatten_json(run_command)
            # update the run command with the new values
            row[1]['command'] = flat
            
            history = row[1]['history']
            data = []
            with io.BytesIO(history) as buffer:
                history = pyarrow.parquet.read_table(pyarrow.PythonFile(buffer, mode="r")).to_pandas()
                data.append(history)
                
            row[1]['history'] = pd.concat(data, ignore_index=True, axis=0)
        # save as a csv
        # experiments.to_csv(fileName, index=False)
        
 
    for i, experiment in experiments.iterrows():
        history = experiment['history']
        history['prunedOut_percentage'] = history['free_parameter_count'] - history['masked_parameter_count']
        experiments.at[i, 'history'] = history
        
    return experiments

# ======== 

def extract_trajectories(parameter_dataset, parameter_indicies, epochs):
    parameter_values = parameter_dataset[parameter_indicies, :]

    trajectories = np.ndarray(shape=(len(parameter_indicies), len(epochs)))
    for i, (epoch, sequence_number) in enumerate(epochs):
        parameters_at_epoch = parameter_values[:, sequence_number]
        trajectories[:, i] = parameters_at_epoch

    return trajectories

def getExperimentWeights(experiement:pd.Series, userName:str, retainedWeights:bool=True):
    """
        Given an experiment from getExperiments, get the experiment, the parameter dataset, and the optimizer datasets
    """
    import h5py as h5
    
    print(f"processing: {experiement['id']}")
    history = experiement.history
    # where retained is true
    retained = history[history['retained'] == retainedWeights]
    # if experimentid already exists, return the filename
    if not (f"{experiement['id']}.h5" in os.listdir("./tempWeights")):
        filename = getWeights(experiement['id'], userName)
    else: 
        print("weights already downloaded")
        filename = f"./tempWeights/{experiement['id']}.h5"
    epochs, parameter_dataset, optimizer_datasets, file = loadWeights(filename, retained)
    return epochs, parameter_dataset, optimizer_datasets, file

