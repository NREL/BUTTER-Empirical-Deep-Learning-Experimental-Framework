# {BUTTER - Empirical Deep Learning Dataset}
 
## Description
 
A brief description of the data including:
- how it was produced?
- why it important/novel
- who/how it might be used
 
This dataset represents an empirical study of the deep learning phenomena on dense fully connected networks, scanning across thirteen datasets, eight network shapes, fourteen depths, twenty-three network sizes (number of trainable parameters), four learning rates, six minibatch sizes, four levels of label noise, and fourteen levels of L1 and L2 regularization each. Multiple repetitions (typically 30, sometimes 10) of each combination of hyperparameters were preformed, and statistics including training and test loss (using a 80% / 20% shuffled train-test split) are recorded at the end of each training epoch. In total, this dataset covers 178 thousand distinct hyperparameter settings ("experiments"), 3.55 million individual training runs (an average of 20 repetitions of each experiments), and a total of 13.3 billion training epochs (three thousand epochs were covered by most runs). Accumulating this dataset consumed 5,448.4 CPU core-years, 17.8 GPU-years, and 111.2 node-years.
 
For each training epoch of each run we recorded 20 performance statistics, and the complete record of these raw values are stored here. For convenience, we additionally include multiple per-experiment summary statistics aggregated over every repetition of that experiment as the "summary" dataset, and also provide several useful slices of the summary data scanning along salient dimensions such as minibatch size and learning rate.
 
 
### Hyperparameter Search Space
 
This is a list of the values of each hyperparameter we swept across in this dataset. **Bold** values are the "core" values that are common to nearly all of the experimental sweeps.
 
+ dataset: Datasets are drawn from the Penn Machine Learning Benchmark Suite (https://epistasislab.github.io/pmlb/)
 + **529_pollen**: regression, 4 features, 1 response variable, 3848 observations
 + **connect_4**: classification, 42 features, 3 classes, 67557 observations
 + **537_houses**: regression, 8 features, 1 response variable, 20640 observations
 + **mnist**: classification, 784 features, 10 classes, 70000 observations
 + **201_pol**: regression, 48 features, 1 response variable, 15000 observations
 + **sleep**: classification, 13 features, 5 classes, 105908 observations
 + **wine_quality_white**: classification, 11 features, 7 classes, 4898 observations
 + nursery: classification, 11 features, 7 classes, 12958 observations
 + adult: classification, 8 features, 4 classes, 4898 observations
 + 505_tecator: regression, 124 features, 1 response variable, 240 observations
 + 294_satellite_image: regression, 36 features, 1 response variable, 6435 observations
 + splice: classification, 60 features, 3 classes, 3188 observations
 + banana: regression, 2 features, 1 response variable, 5300 observations
+ shape: Network shape, which in combination with depth and size determines the neural network topology used.
 + **rectangle**: All layers except for the output layer have the same width.
 + **trapezoid**: Layer widths linearly decrease from input to output.
 + **exponential**: Layer widths exponentially decay from input to output.
 + **wide_first_2x**: Like rectangle, but the first layer is twice as wide as the other layers.
 + rectangle_residual: rectangle but with residual connections for every layer.
 + wide_first_4x: Like rectangle, but the first layer is four times as wide as the other layers.
 + wide_first_8x: Like rectangle, but the first layer is eight times as wide as the other layers.
 + wide_first_16x: Like rectangle, but the first layer is sixteen times as wide as the other layers.
+ size: {**2^5, 2^6, ... 2^24**,  2^25, 2^26, 2^27}
 + The approximate number of trainable parameters in the network. The exact number is recorded in the dataset. If a topology of the desired shape and depth could not be found with a number of trainable parameters within 20% of the size setting, the experiment may not be included. Typically this occurs with only the smallest few size settings. The exact number of trainable parameters is reported in the dataset as num_free_parameters, layer widths are reported in the widths column, and the exact topology is reported in the network_structure column.
+ depth: {**2**,**3**,**4**,**5**,6,**7**,**8**,**9**,**10**, 12, 14, 16, 18, 20}
 + the number of layers in the network
+ learning rate: {0.01, 0.001, **0.0001**, 0.00001}
+ minibatch size: {32, 64, 128, **256**, 512, 1024}
+ L1 and L2 regularization penalties: {**0.0**, .32, .16, .08, .04, .02, .01, .005, .0025, .00125, .000625, .0003125, .00015625, 7.8125e-5}
+ label noise level: {**0.0**, 0.05, .1, .15, .2}
+ training epochs: {300, **3000**, 30000}
+ repetitions: {10, **30**}
 + The number of times each identical experiment was repeated with different random seeds.
+ optimizer: **ADAM**
+ hidden layer activation function: **ReLU**
+ train-test split: **80% training set, 20% test set**
 
 
 
For regressions, we used mean squared error as the loss function; for binary classifications, we used binary cross entropy; and for all other classifications, we used categorical cross entropy loss and one-hot encoding.
 
### Hyperparameter Sweeps
 
Sweeping the entire hypercube of eight hyperparameter dimensions would require evaluation of over 90 million distinct experiments. To reduce the total number of experiments executed while preserving significant utility of the dataset, we conducted seven overlapping hyperparameter sweeps, each covering a hypercube (or in one case two hypercubes) of the hyperparameter search space:
 
+ primary sweep: The primary sweep tests all datasets and depths with all but the largest sizes without regularization or label noise at the core learning rate and batch size.
 + typical repetitions: **30**
 + datasets: {**529_pollen, connect_4, 537_houses, mnist, 201_pol, sleep, wine_quality_white**, nursery, adult, 505_tecator, 294_satellite_images, splice, banana}
 + shapes: {**rectangle, trapezoid, exponential, wide_first_2x**, rectangle_residual, wide_first_4x, wide_first_8x, wide_first_16x}
 + size: {**2^5, 2^6, ... 2^24**, 2^25}
 + depth: {**2**,**3**,**4**,**5**,6,**7**,**8**,**9**,**10**, 12, 14, 16, 18, 20}
 + learning rate: **.0001**
 + minibatch size: **256**
 + regularization: **none**
 + label noise: **0**
 
+ 300 epoch sweep: This sweep extends the primary sweep to larger sizes for 300 training epochs.
 + typical repetitions: 10 for rectangle, 0-10 for other shapes
 + datasets: {**529_pollen, connect_4, 537_houses, mnist, 201_pol, sleep, wine_quality_white**}
 + shapes: {**rectangle, trapezoid, exponential, wide_first_2x**}
 + size: {2^25, 2^26, 2^27}
 + depth: {**2**,**3**,**4**,**5**,6,**7**,**8**,**9**,**10**, 12, 14, 16, 18, 20}
 + learning rate: **.0001**
 + minibatch size: **256**
 + regularization: **none**
 + label noise: **0**
 
+ 30k epoch sweep: This sweep extends the primary sweep to 30 thousand training epochs for smaller sizes.
 + typical repetitions: 10 for rectangle, 0-10 for other shapes
 + datasets: {**529_pollen, connect_4, 537_houses, mnist, 201_pol, sleep, wine_quality_white**}
 + shapes: {**rectangle, trapezoid, exponential, wide_first_2x**}
 + size: {**2^5, 2^6 .. 2^18**}
 + depth: {**2**,**3**,**4**,**5**,6,**7**,**8**,**9**,**10**, 12, 14, 16, 18, 20}
 + learning rate: **.0001**
 + minibatch size: **256**
 + regularization: **none**
 + label noise: **0**
 
+ learning rate sweep: This sweep scans the core shapes over several learning rates.
 + typical repetitions: **30**
 + datasets: {**529_pollen, connect_4, 537_houses, mnist, 201_pol, sleep, wine_quality_white**, nursery, adult}
 + shapes: {**rectangle, trapezoid, exponential, wide_first_2x**}
 + size: {**2^5, 2^6, ... 2^24**}
 + depth: {**2**,**3**,**4**,**5**,**7**,**8**,**9**,**10**, 12, 14, 16, 18, 20}
 + learning rate: {0.01, 0.001, **0.0001**, 0.00001}
 + minibatch size: **256**
 + regularization: **none**
 + label noise: **0**
 + label noise sweep: This sweep scans the core shapes over several label noise levels at two different learning rates.
 + typical repetitions: **30**
 + datasets: {**529_pollen, connect_4, 537_houses, mnist, 201_pol, sleep, wine_quality_white**}
 + shapes: {**rectangle, trapezoid, exponential, wide_first_2x**}
 + size: {**2^5, 2^6, ... 2^24**}
 + depth: {**2**,**3**,**4**,**5**,**7**,**8**,**9**,**10**, 12, 14, 16, 18, 20}
 + learning rate: {0.001, **0.0001**}
 + minibatch size: **256**
 + regularization: **none**
 + label noise:
   + {**0**, 0.05} at learning rate 0.001
   + {**0.0**, 0.05, .1, .15, .2} at learning rate **0.0001**
 
+ batch size sweep: This sweep scans rectangular networks of the core depths over several batch sizes.
 + typical repetitions: **30**
 + datasets: {**529_pollen, connect_4, 537_houses, mnist, 201_pol, sleep, wine_quality_white**, nursery, adult}
 + shapes: {**rectangle, trapezoid, exponential, wide_first_2x**}
 + size: {**2^5, 2^6, ... 2^24**}
 + depth: {**2**,**3**,**4**,**5**,**7**,**8**,**9**,**10**}
 + learning rate: **0.0001**
 + minibatch size: {32, 64, 128, **256**, 512, 1024}
 + regularization: **none**
 + label noise: **0**
 
+ regularization sweep: This sweep scans rectangular networks of the core depths over several L1 and L2 regularization levels.
 + typical repetitions: **30**
 + datasets: {**529_pollen, connect_4, 537_houses, mnist, 201_pol, sleep, wine_quality_white**, nursery, adult}
 + shapes: {**rectangle, trapezoid, exponential, wide_first_2x**}
 + size: {**2^5, 2^6, ... 2^24**, 2^25}
 + depth: {**2**,**3**,**4**,**5**,**7**,**8**,**9**,**10**}
 + learning rate: **0.0001**
 + minibatch size: **256**
 + regularization:
   + L1: {**0.0**, .32, .16, .08, .04, .02, .01, .005, .0025, .00125, .000625, .0003125, .00015625, 7.8125e-5}
   + L2: {**0.0**, .32, .16, .08, .04, .02, .01, .005, .0025, .00125, .000625, .0003125, .00015625, 7.8125e-5}
 + label noise: **0**
 
 
## Directory structure
 
If the dataset is made up of multiple files a description of how they are/will
be stored in relation to each other.
 
## Data Format
 
The complete raw dataset is available in the /all_runs/ partitioned parquet dataset. Each row in this dataset is a record of one training run of a network. Several statistics were recorded at the end of each training epoch, and those records are stored in this row as arrays indexed by training epoch. For convenience, we also provide the /complete_summary/ partitioned parquet dataset which contains statistics aggregated over all repetitions of the same experiment including average and median test and training losses at the end of each training epoch. Distinct experiments are uniquely and consistently identified in both datasets by an 'experiment_id'. Additionally, we have created separate raw run and summary datasets for each experimental sweep so that they can be downloaded and queried separately if the entire dataset is not needed. _The schemas of summary and run datasets are the same for every sweep._
 
### File Hierarchy and Descriptions
 
/all_runs/ contains all of the experimental run records for all sweeps
/complete_summary/ contains per-experiment statistics aggregated over every run of each distinct experiment for all sweeps
/complete_summary.tar is a tarball of /experiment_summary/
 
/primary_sweep_runs/ contains all of the experimental run records for the primary sweep
/primary_sweep_summary/ contains summary experiment statistics for the primary sweep
/primary_sweep_summary.tar is a tarball of /primary_sweep_summary/
 
/learning_rate_sweep_runs/ contains all of the experimental run records for the learning rate sweep
/learning_rate_sweep_summary/ contains summary experiment statistics for the learning rate sweep
/learning_rate_sweep_summary.tar is a tarball of /learning_rate_sweep_summary/
 
/label_noise_sweep_runs/ contains all of the experimental run records for the label noise sweep
/label_noise_sweep_summary/ contains summary experiment statistics for the label noise sweep
/label_noise_sweep_summary.tar is a tarball of /label_noise_sweep_summary/
 
/batch_size_sweep_runs/ contains all of the experimental run records for the batch size sweep
/batch_size_sweep_summary/ contains summary experiment statistics for the batch size sweep
/batch_size_sweep_summary.tar is a tarball of /batch_size_sweep_summary/
 
/regularization_sweep_runs/ contains all of the experimental run records for the regularization sweep
/regularization_sweep_summary/ contains summary experiment statistics for the regularization sweep
/regularization_sweep_summary.tar is a tarball of /regularization_sweep_summary/
 
/300_epoch_sweep_runs/ contains all of the experimental run records for the 300 epoch sweep
/300_epoch_sweep_summary/ contains summary experiment statistics for the 300 epoch sweep
/300_epoch_sweep_summary.tar is a tarball of /300_epoch_sweep_summary/
 
/30k_epoch_sweep_runs/ contains all of the experimental run records for the 30k epoch sweep
/30k_epoch_sweep_summary/ contains summary experiment statistics for the 30k epoch sweep
/30k_epoch_sweep_summary.tar is a tarball of /30k_epoch_sweep_summary/
 
### Experiment Summary Schema
 
For preliminary analysis, we recommend using the summary dataset as it is smaller and more convenient to work with than the full run dataset. However, the entire record of every repetition of every experiment is stored in the run dataset, allowing other statistics to be computed from the raw data. Each experiment has a unique experiment_id value, which matches run records in the runs dataset. Summary data is partitioned by dataset, shape, learning rate, batch size, kernel regularizer, label noise, depth, and number of training epochs.
 
#### The columns of the summary dataset are:
 
+ experiment_id the unique id for this experiment
+ primary_sweep: bool, true iff this experiment is part of the primary sweep
+ 300_epoch_sweep: bool, true iff this experiment is part of the 300 epoch sweep
+ 30k_epoch_sweep: bool, true iff this experiment is part of the 30k epoch sweep
+ learning_rate_sweep: bool, true iff this experiment is part of the learning rate sweep
+ label_noise_sweep: bool, true iff this experiment is part of the label noise sweep
+ batch_size_sweep: bool, true iff this experiment is part of the batch size sweep
+ regularization_sweep: bool, true iff this experiment is part of the regularization sweep
+ activation: string, the activation function used for hidden layers
+ batch: string, a nickname for the experimental batch this experiment belongs to
+ batch_size: uint32, minibatch size
+ dataset: string, name of the dataset used
+ depth: uint8, number of layers
+ early_stopping: string, early stopping policy
+ epochs: uint32, number of training epochs in this run
+ input_activation: string, input activation function
+ kernel_regularizer: string, null if no regularizer is used
+ kernel_regularizer.l1: float32, L1 regularization penalty coefficient
+ kernel_regularizer.l2: float32, L2 regularization penalty coefficient
+ kernel_regularizer.type: string, name of kernel regularizer used (null if none used)
+ label_noise: float32, amount of label noise applied to dataset before training (.05 means 5% label noise)
+ learning_rate: float32, learning rate used
+ optimizer: string, name of the optimizer used
+ output_activation: string, activation function for output layer
+ shape: string, network shape
+ size: uint64, approximate number of trainable parameters used
+ task: string, name of training task
+ test_split: float32, test split proportion
+ test_split_method: string, test split method
+ num_free_parameters: uint64, exact number of trainable parameters
+ widths: [uint32], list of layer widths used
+ network_structure: string, marshaled json representation of network structure used
+ num_runs: uint8, number of runs aggregated in this summary record
+ num: [uint8], number of runs aggregated in this summary record at each epoch
+ test_loss_num_finite: [uint8], number of finite test losses at each epoch
+ test_loss_avg: [float32], average test loss at each epoch
+ test_loss_stddev: [float32], standard deviation of the test loss at each epoch
+ test_loss_min: [float32], minimum test loss at each epoch
+ test_loss_max: [float32], maximum test loss at each epoch
+ test_loss_median: [float32], median test loss at each epoch
+ train_loss_num_finite: [uint8], number of finite training losses at each epoch
+ train_loss_avg: [float32], average training loss at each epoch
+ train_loss_stddev: [float32], standard deviation of training losses at each epoch
+ train_loss_min: [float32], minimum training loss at each epoch
+ train_loss_max: [float32], maximum training loss at each epoch
+ train_loss_median: [float32], median training loss at each epoch
+ test_accuracy_avg: [float32], average test accuracy at each epoch
+ test_accuracy_stddev: [float32], test accuracy standard deviation accuracy at each epoch
+ test_accuracy_median: [float32], median test accuracy at each epoch
+ train_accuracy_avg: [float32], average training accuracy at each epoch
+ train_accuracy_stddev: [float32], standard deviation of the training accuracy at each epoch
+ train_accuracy_median: [float32], median training accuracy at each epoch
+ test_mean_squared_error_avg: [float32], test MSE at each epoch
+ test_mean_squared_error_stddev: [float32], test MSE standard deviation at each epoch
+ test_mean_squared_error_median: [float32], median test MSE at each epoch
+ train_mean_squared_error_avg: [float32], average training MSE at each epoch
+ train_mean_squared_error_stddev: [float32], training MSE standard deviation at each epoch
+ train_mean_squared_error_median: [float32], median training MSE at each epoch
+ test_kullback_leibler_divergence_avg: [float32], average test KL-Divergence at each epoch
+ test_kullback_leibler_divergence_stddev: [float32], test KL-Divergence standard deviation at each epoch
+ test_kullback_leibler_divergence_median: [float32], median test KL-Divergence at each epoch
+ train_kullback_leibler_divergence_avg: [float32], average training KL-Divergence at each epoch
+ train_kullback_leibler_divergence_stddev: [float32], training KL-Divergence standard deviation at each epoch
+ train_kullback_leibler_divergence_median: [float32], median training KL-Divergence at each epoch
 
### Run Schema
 
Each row of the run dataset represents a single training run. Each training run was instrumented to record various statistics at the end of each training epoch, and those statistics are stored as epoch-indexed arrays in each row. Runs are labeled with an 'experiment_id' which was used to aggregate repetitions of the same experimental parameters together in the summary dataset. An experiment_id in the experiment dataset correspond to the same experiment_id in the summary dataset.
 
#### The columns of the run dataset are:
 
+ experiment_id: uint32, id of the experiment this run was a repetition of
+ run_id: string, unique id of this run
+ primary_sweep: bool, true iff this experiment is part of the primary sweep
+ 300_epoch_sweep: bool, true iff this experiment is part of the 300 epoch sweep
+ 30k_epoch_sweep: bool, true iff this experiment is part of the 30k epoch sweep
+ learning_rate_sweep: bool, true iff this experiment is part of the learning rate sweep
+ label_noise_sweep: bool, true iff this experiment is part of the label noise sweep
+ batch_size_sweep: bool, true iff this experiment is part of the batch size sweep
+ regularization_sweep: bool, true iff this experiment is part of the regularization sweep
+ activation: string, the activation function used for hidden layers
+ batch: string, a nickname for the experimental batch this experiment belongs to
+ batch_size: uint32, minibatch size
+ dataset: string, name of the dataset used
+ depth: uint8, number of layers
+ early_stopping: string, early stopping policy
+ epochs: uint32, number of training epochs in this run
+ input_activation: string, input activation function
+ kernel_regularizer: string, null if no regularizer is used
+ kernel_regularizer.l1: float32, L1 regularization penalty coefficient
+ kernel_regularizer.l2: float32, L2 regularization penalty coefficient
+ kernel_regularizer.type: string, name of kernel regularizer used (null if none used)
+ label_noise: float32, amount of label noise applied to dataset before training (.05 means 5% label noise)
+ learning_rate: float32, learning rate used
+ optimizer: string, name of the optimizer used
+ output_activation: string, activation function for output layer
+ python_version: string, python version used
+ shape: string, network shape
+ size: uint64, approximate number of trainable parameters used
+ task: string, name of training task
+ task_version: uint16, major version of training task
+ tensorflow_version: string, tensorflow version
+ test_split: float32, test split proportion
+ test_split_method: string, test split method
+ num_free_parameters: uint64, exact number of trainable parameters
+ widths: [uint32], list of layer widths used
+ network_structure: string, marshaled json representation of network structure used
+ platform: string, platform run executed on
+ git_hash: string, git hash of version of experimental framework used
+ hostname: string, hostname of system that executed this run
+ seed: int64, random seed used to initialize this run
+ start_time: int64, unix timestamp of start time of this run
+ update_time: int64, unix timestamp of completion time of this run
+ command: string, complete marshaled json representation of this run's settings
+ network_structure: string, marshaled json representation of this run's network configuration
+ widths: uint32, list of layer widths for the network used
+ num_free_parameters: uint64, exact number of free parameters in the tested network
+ val_loss: [float32], test loss at the end of each epoch
+ loss: [float32], training loss at the end of each epoch
+ val_accuracy: [float32], test accuracy at the end of each epoch
+ accuracy: [float32], training accuracy at the end of each epoch
+ val_mean_squared_error: [float32], test MSE at the end of each epoch
+ mean_squared_error: [float32], training MSE at the end of each epoch
+ val_mean_absolute_error: [float32], test MAE at the end of each epoch
+ mean_absolute_error: [float32], training MAE at the end of each epoch
+ val_root_mean_squared_error: [float32], test RMS error at the end of each epoch
+ root_mean_squared_error: [float32], training RMS error at the end of each epoch
+ val_mean_squared_logarithmic_error: [float32], test MSLE at the end of each epoch
+ mean_squared_logarithmic_error: [float32], training MSLE at the end of each epoch
+ val_hinge: [float32], test hinge loss at the end of each epoch
+ hinge: [float32], training hinge loss at the end of each epoch
+ val_squared_hinge: [float32], test squared hinge loss at the end of each epoch
+ squared_hinge: [float32], training squared hinge loss at the end of each epoch
+ val_cosine_similarity: [float32], test cosine similarity at the end of each epoch
+ cosine_similarity: [float32], training cosine similarity at the end of each epoch
+ val_kullback_leibler_divergence: [float32], test KL-divergence at the end of each epoch
+ kullback_leibler_divergence: [float32], training KL-divergence at the end of each epoch
 
 
## Code Examples
 
Example scripts of how to access the data IN THE CLOUD. A jupyter notebook or
link to a github repo with examples can be used instead.
 
## References
 
Any helpful references other documentation
 
## Disclaimer and Attribution
 
Optional additional attributes/disclaimers


