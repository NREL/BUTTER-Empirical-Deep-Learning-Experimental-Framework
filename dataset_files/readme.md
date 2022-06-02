# {BUTTER - Empirical Deep Learning Dataset}

## Description

A brief description of the data including:
- how it was produced?
- why it important/novel
- who/how it might be used

## Directory structure

If the dataset is made up of multiple files a description of how they are/will
be stored in relation to each other.

## Data Format

The complete raw dataset is avaliable in the /all_runs/ partitioned parquet dataset. Each row in this dataset is a record of one training run of a network. Several statistics were recorded at the end of each training epoch, and those records are stored in this row as arrays indexed by training epoch. For convience, we also provide the /complete_summary/ partitioned parquet dataset which contains statistics aggregated over all repetitions of the same experiment including average and median test and training losses at the end of each training epoch. Distinct experiments are uniquely and consistently identified in both datasets by an 'experiment_id'. Additionally, we have created separate raw run and summary datasets for each experimental sweep so that they can be downloaded and queried separately if the entire dataset is not needed. _The schemas of summary and run datasets are the same for every sweep._


/all_runs/ contains all of the experimental run records for all sweeps
/complete_summary/ contains per-experiment statistics aggregated over every run of each distinct experiment for all sweeps
/complete_summary.tar is a tarball of /experiment_summary/

/primary_sweep_runs/ contains all of the experimental run records for the primary sweep
/primary_sweep_summary/ contains summary experiment statistics for the primary sweep

/learning_rate_sweep_runs/ contains all of the experimental run records for the learning rate sweep
/learning_rate_sweep_summary/ contains summary experiment statistics for the learning rate sweep

/label_noise_sweep_runs/ contains all of the experimental run records for the label noise sweep
/label_noise_sweep_summary/ contains summary experiment statistics for the label noise sweep

/batch_size_sweep_runs/ contains all of the experimental run records for the batch size sweep
/batch_size_sweep_summary/ contains summary experiment statistics for the batch size sweep

/regularization_sweep_runs/ contains all of the experimental run records for the regularization sweep
/regularization_sweep_summary/ contains summary experiment statistics for the regularization sweep

/300_epoch_sweep_runs/ contains all of the experimental run records for the 300 epoch sweep
/300_epoch_sweep_summary/ contains summary experiment statistics for the 300 epoch sweep

/30k_epoch_sweep_runs/ contains all of the experimental run records for the 30k epoch sweep
/30k_epoch_sweep_summary/ contains summary experiment statistics for the 30k epoch sweep

### Experiment Summary Schema

For preliminary analysis, we recommend using the summary dataset as it is smaller and more convenient to work with than the full run dataset. However, the entire record of every repetition of every experiment is stored in the run dataset, allowing other statistics to be computed from the raw data. Each experiment has a unique experiment_id value, which matches run records in the runs dataset. Summary data is partitioned by dataset, shape, learning rate, batch size, kernel regularizer, label noise, depth, and number of training epochs. 

The columns of the summary dataset are:

- dataset/variable/column names
- units

## Code Examples

Example scripts of how to access the data IN THE CLOUD. A jupyter notebook or
link to a github repo with examples can be used instead.

## References

Any helpful references other documentation

## Disclaimer and Attribution

Optional additional attributes/disclaimers