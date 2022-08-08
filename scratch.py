Begin chunk [2085870].
{'activation': ['relu'],
 'batch': ['fixed_3k_1'],
 'batch_size': [256],
 'dataset': ['201_pol'],
 'depth': [3],
 'epochs': [3000],
 'input_activation': ['relu'],
 'kernel_regularizer': [None],
 'kernel_regularizer_l1': [None],
 'kernel_regularizer_l2': [None],
 'kernel_regularizer_type': [None],
 'label_noise': [0.0],
 'learning_rate': [1e-05],
 'optimizer': ['adam'],
 'momentum': [None],
 'nesterov': [None],
 'output_activation': [None],
 'shape': ['rectangle'],
 'size': [16777216],
 'task': ['AspectTestTask'],
 'test_split': [0.2],
 'experiment_id': [2085870],
 'primary_sweep': [False],
 '300_epoch_sweep': [False],
 '30k_epoch_sweep': [False],
 'learning_rate_sweep': [True],
 'label_noise_sweep': [False],
 'batch_size_sweep': [False],
 'regularization_sweep': [False],
 'learning_rate_batch_size_sweep': [False],
 'size_adjusted_regularization_sweep': [False],
 'optimizer_sweep': [False],
 'num_free_parameters': [16780943],
 'num_runs': [10]}
End chunk [2085870].
Stored 1 in chunk [2085870],
 280 / 251101.
Begin chunk [2171641].
{'activation': ['relu'],
 'batch': ['optimizer_1'],
 'batch_size': [256],
 'dataset': ['201_pol'],
 'depth': [3],
 'epochs': [3000],
 'input_activation': ['relu'],
 'kernel_regularizer': [None],
 'kernel_regularizer_l1': [None],
 'kernel_regularizer_l2': [None],
 'kernel_regularizer_type': [None],
 'label_noise': [0.0],
 'learning_rate': [1e-05],
 'optimizer': ['SGD'],
 'momentum': [0.0],
 'nesterov': [False],
 'output_activation': ['softmax'],
 'shape': ['rectangle'],
 'size': [128],
 'task': ['AspectTestTask'],
 'test_split': [0.2],
 'experiment_id': [2171641],
 'primary_sweep': [False],
 '300_epoch_sweep': [False],
 '30k_epoch_sweep': [False],
 'learning_rate_sweep': [False],
 'label_noise_sweep': [False],
 'batch_size_sweep': [False],
 'regularization_sweep': [False],
 'learning_rate_batch_size_sweep': [False],
 'size_adjusted_regularization_sweep': [False],
 'optimizer_sweep': [True],
 'num_free_parameters': [137],
 'num_runs': [29]}
multiprocess.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/projects/dmpapps/ctripp/env/dmp-gpu2/lib/python3.10/site-packages/multiprocess/pool.py",
 line 125,
 in worker
    result = (True,
 func(*args,
 **kwds))
  File "/projects/dmpapps/ctripp/env/dmp-gpu2/lib/python3.10/site-packages/pathos/helpers/mp_helper.py",
 line -1,
 in <lambda>
  File "/lustre/eaglefs/projects/dmpapps/ctripp/src/dmp/util/extract_summary_parquet.py",
 line 649,
 in download_chunk
    parquet.write_to_dataset(
  File "pyarrow/table.pxi",
 line 3625,
 in pyarrow.lib.Table.from_pydict
  File "pyarrow/table.pxi",
 line 5167,
 in pyarrow.lib._from_pydict
  File "pyarrow/array.pxi",
 line 342,
 in pyarrow.lib.asarray
  File "pyarrow/array.pxi",
 line 316,
 in pyarrow.lib.array
  File "pyarrow/array.pxi",
 line 39,
 in pyarrow.lib._sequence_to_array
  File "pyarrow/error.pxi",
 line 144,
 in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi",
 line 123,
 in pyarrow.lib.check_status
pyarrow.lib.ArrowTypeError: Expected bytes,
 got a 'float' object
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/projects/dmpapps/ctripp/env/dmp-gpu2/lib/python3.10/runpy.py",
 line 196,
 in _run_module_as_main
Begin chunk [2171644].
    return _run_code(code,
 main_globals,
 None,

  File "/projects/dmpapps/ctripp/env/dmp-gpu2/lib/python3.10/runpy.py",
 line 86,
 in _run_code
    exec(code,
 run_globals)
  File "/lustre/eaglefs/projects/dmpapps/ctripp/src/dmp/util/extract_summary_parquet.py",
 line 702,
 in <module>
    main()
  File "/lustre/eaglefs/projects/dmpapps/ctripp/src/dmp/util/extract_summary_parquet.py",
 line 692,
 in main
    for num_rows,
 chunk in results:
  File "/projects/dmpapps/ctripp/env/dmp-gpu2/lib/python3.10/site-packages/multiprocess/pool.py",
 line 870,
 in next
{'activation': ['relu'],
 'batch': ['optimizer_1'],
 'batch_size': [256],
 'dataset': ['201_pol'],
 'depth': [3],
 'epochs': [3000],
 'input_activation': ['relu'],
 'kernel_regularizer': [None],
 'kernel_regularizer_l1': [None],
 'kernel_regularizer_l2': [None],
 'kernel_regularizer_type': [None],
 'label_noise': [0.0],
 'learning_rate': [1e-05],
 'optimizer': ['SGD'],
 'momentum': [0.0],
 'nesterov': [False],
 'output_activation': ['softmax'],
 'shape': ['rectangle'],
 'size': [512],
 'task': ['AspectTestTask'],
 'test_split': [0.2],
 'experiment_id': [2171644],
 'primary_sweep': [False],
 '300_epoch_sweep': [False],
 '30k_epoch_sweep': [False],
 'learning_rate_sweep': [False],
 'label_noise_sweep': [False],
 'batch_size_sweep': [False],
 'regularization_sweep': [False],
 'learning_rate_batch_size_sweep': [False],
 'size_adjusted_regularization_sweep': [False],
 'optimizer_sweep': [True],
 'num_free_parameters': [501],
 'num_runs': [30]}
    raise value
