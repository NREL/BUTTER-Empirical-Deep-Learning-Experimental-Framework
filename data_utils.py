import pandas as pd
import pyarrow
import pyarrow.parquet as parquet
import os
import numpy as np

def butter_data_path():
    return os.getenv("DMP_BUTTER_DATA_DIR", "s3://oedi-data-lake/butter")

def get_s3_filesystem():
    import s3fs
    import botocore
    return s3fs.S3FileSystem(config_kwargs={"signature_version": botocore.UNSIGNED})

partition_cols = [
    'dataset',
    'learning_rate',
    'batch_size',
    'kernel_regularizer.type',
    'label_noise',
    'epochs',
    'shape',
    'depth',
]

def read_pandas(sweep, filters, columns):
    # Get path from environment. Default is OEDI data lake.
    path = butter_data_path()
    
    # Clean up path (trim filesystem identifier) and create filesystem objects
    if path[:5]=="s3://":
        fs = get_s3_filesystem()
        path = path[5:] + "/" + sweep
        schema_fp = fs.open(f'{path}/_common_metadata')
    else:
        fs = None
        path = path+"/"+sweep
        schema_fp = open(f'{path}/_common_metadata', 'rb')
    
    # Read schema data from _common_metadata file
    schema = parquet.read_schema(schema_fp)
    schema_fp.close()

    # Read parquet table
    table = parquet.read_table(path, filesystem=fs, filters=filters, schema=schema, columns=columns)

    # Convert to pandas
    return table.to_pandas()

def extract_data(original_df, downsample_epochs=False, grouper='shape'):

    original_df['epoch'] = original_df['test_loss_median'].apply(lambda x: list(range(1, 1 + len(x))))

    epoch_wise_values = ["test_loss_median", "epoch"]

    df = original_df.explode(epoch_wise_values)

    for value in epoch_wise_values:
        df[value] = df[value].astype("float32")
    df["epoch"] = df["epoch"].astype("int")

    #downsample by epoch logarithmically
    if downsample_epochs:
        keep_epochs = np.unique(np.logspace(0,np.log10(3000),100,base=10).astype(int))
        df = df[df["epoch"].isin(keep_epochs)]

    # num_free_parameters is not unique in sizes. Take first experiment
    df = df.sort_values('experiment_id').groupby(list(set(['dataset',grouper,'num_free_parameters','epoch']))).first().reset_index()
    return df