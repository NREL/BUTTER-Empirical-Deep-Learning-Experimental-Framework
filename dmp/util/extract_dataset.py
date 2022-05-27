from multiprocessing import Pool
import os

import h5py
from jobqueue.connect import connect
from jobqueue.cursor_manager import CursorManager
import numpy

from dmp.logging.postgres_parameter_map import PostgresParameterMap


def main():
    credentials = connect.load_credentials('dmp')
    parameter_map = PostgresParameterMap(cursor)

    # base_path = '/projects/dmpapps/jperrsau/datasets/2022_05_20_fixed_3k_1/'
    base_path = '/home/ctripp/scratch/'
    file_name = os.path.join(base_path, 'fixed_3k_1.hdf5')

    # fixed_3k_1_meta.csv.gz

    fixed_parameters = [
        ('batch', 'fixed_3k_1'),
    ]

    variable_parameter_kinds = [
        'dataset',
        'learning_rate',
        'label_noise',
        'depth',
        'shape',
        'size',
    ]

    # patameters_metadata.to_csv(base_path+'patameters_metadata.csv.gz',
    #                        index=False, compression='gzip')

    # fixed_3k_1_meta = experiment_table_meta.query('batch=='fixed_3k_1'')
    # fixed_3k_1_meta.to_csv(base_path+'fixed_3k_1_meta.csv.gz',
    #                        index=False, compression='gzip')

    # fixed_3k_1_dataset_...csv.gz
    return_values = [
        ('num', 'uint8'),
        ('val_loss_num_finite', 'uint8'),
        ('val_loss_avg', 'float32'),
        ('val_loss_stddev', 'float32'),
        ('val_loss_min', 'float32'),
        ('val_loss_max', 'float32'),
        ('loss_num_finite', 'uint8'),
        ('loss_avg', 'float32'),
        ('loss_stddev', 'float32'),
        ('loss_min', 'float32'),
        ('loss_max', 'float32'),
        ('loss_median', 'float32'),
        ('val_accuracy_avg', 'float32'),
        ('val_accuracy_stddev', 'float32'),
        ('val_accuracy_median', 'float32'),
        ('val_loss_median', 'float32'),
        ('accuracy_avg', 'float32'),
        ('accuracy_stddev', 'float32'),
        ('accuracy_median', 'float32'),
        ('val_mean_squared_error_avg', 'float32'),
        ('val_mean_squared_error_stddev', 'float32'),
        ('val_mean_squared_error_median', 'float32'),
        ('mean_squared_error_avg', 'float32'),
        ('mean_squared_error_stddev', 'float32'),
        ('mean_squared_error_median', 'float32'),
        ('val_kullback_leibler_divergence_avg', 'float32'),
        ('val_kullback_leibler_divergence_stddev', 'float32'),
        ('val_kullback_leibler_divergence_median', 'float32'),
        ('kullback_leibler_divergence_avg', 'float32'),
        ('kullback_leibler_divergence_stddev', 'float32'),
        ('kullback_leibler_divergence_median', 'float32'),
    ]

    # dataset_name = '3k_data'
    data_group_name = '3k'
    num_epochs = 3000
    parameters_group_name = 'parameters'

    all_kinds = list((p[0] for p in fixed_parameters)) + \
        variable_parameter_kinds

    dimensions = []
    parameter_index = {}
    parameters = []
    for dim, kind in enumerate(all_kinds):
        indexes = []
        dimensions.append(indexes)
        for i, (value, id) in enumerate(sorted(parameter_map.get_all_parameters_for_kind(kind))):
            parameter_data = (kind, value, dim, i, id)
            indexes.append(parameter_data)
            parameters.append(parameter_data)
            parameter_index[id] = parameter_data

    # patameters_metadata = pd.DataFrame(
    #     parameters, columns=['kind', 'value', 'dimension', 'index', 'id'])

    # patameters_metadata.to_csv(base_path+'patameters_metadata.csv',
    #                            index=False)

    shape = [len(indexes) for indexes in dimensions] + [num_epochs]
    with h5py.File(file_name, 'w') as f:
        group = f.create_group(data_group_name)
        group.attrs['num_epochs'] = num_epochs

        parameters_group = group.create_group(parameters_group_name)

        def make_parameter_dataset(name, dtype, index, description):
            dataset = parameters_group.create_dataset(
                name, (len(parameters)), dtype=dtype)
            dataset[:] = np.array(
                [p[index] for p in parameters],
            )
            dataset.attrs['description'] = description

        make_parameter_dataset('kind', h5py.special_dtype(
            vlen=str), 0, 'What does this parameter define?')
        make_parameter_dataset('value', h5py.special_dtype(
            vlen=str), 1, 'What value of this parameter\'s kind does this parameter indicate?')
        make_parameter_dataset(
            'dimension', 'uint8', 2, 'What dimension of data datasets indexes this parameter kind?')
        make_parameter_dataset(
            'index', 'uint8', 3, 'What index of of the dimension indicates this parameter value?')

        data_group = group.create_group('data')
        for variable, dtype in return_values:
            ds = data_group.create_dataset(
                variable,
                tuple(shape),
                chunks=tuple(([1] * (len(shape) - 1)) + [num_epochs]),
                dtype=dtype,
                compression='lzf',
            )

            ds.attrs['variable'] = variable
            if dtype == 'float32':
                ds = numpy.NaN
            else:
                ds = 0

        experiment_ids = []
        with CursorManager(credentials) as cursor:
            cursor.execute(f'''
        SELECT
            experiment_id
        FROM
            experiment_summary_ s
        WHERE
            s.experiment_parameters @> (array[ {parameter_map.to_parameter_ids(fixed_parameters)} ])::smallint[]
        ;''')

            for row in cursor.fetchall():
                experiment_ids.append(row[0])

        chunk_size = 64
        chunks = [
            list(experiment_ids[chunk_size*chunk:chunk_size*(chunk+1)])
            for chunk in range(int(np.ceil(len(experiment_ids) / chunk_size)))]

        def download_chunk(chunk):
            # print(f'Begin chunk {chunk[0]}.')
            result_block = []
            # with h5py.File(file_name, 'a') as f:
            #     data_group = f[data_group_name]['data']
            with CursorManager(credentials) as cursor:
                for variable, dtype in return_values:
                    cursor.execute(f'''
    SELECT 
        experiment_id, experiment_parameters, {variable}
    FROM
        experiment_summary_ s
    WHERE
        s.experiment_id in ( {','.join((str(eid) for eid in chunk))} )
    ;''')
                    for row in cursor.fetchall():
                        dims = [0] * len(all_kinds)
                        for id in row[1]:
                            if id in parameter_index:
                                kind, value, dim, i, id = parameter_index[id]
                                dims[dim] = i
                        data = np.array(row[2], dtype=dtype)[0:num_epochs]
                        result_block.append((variable, dims, data))

            # print(f'End chunk {chunk[0]}.')
            return result_block

        print(f'Created {len(chunks)} chunks.')
        import pathos.multiprocessing as multiprocessing

        # SchemaUpdate(credentials, logger, chunks[0])
        data_group = f[data_group_name]['data']
        results = None

        num_stored = 0
        with multiprocessing.ProcessPool(32) as pool:
            results = pool.uimap(download_chunk, chunks)
            for result_block in results:
                num_stored += 1
                print(f'Storing chunk {num_stored} / {len(chunks)}...')
                for variable, dims, data in result_block:
                    dataset = data_group[variable]
                    dataset[dims[0], dims[1], dims[2], dims[3],
                            dims[4], dims[5], dims[6]] = data

    print('Done.')


if __name__ == "__main__":
    main()
