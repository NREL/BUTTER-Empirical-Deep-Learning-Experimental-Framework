import sys
import datetime
from joblib import Memory
from ipywidgets import interact, interact_manual
import inspect
import scipy



import pandas as pd
import numpy as np

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import matplotlib.pyplot as plt

from psycopg2 import sql

import scipy.interpolate
import colorsys

from jobqueue.connect import connect

credentials = connect.load_credentials('dmp')
memory = Memory(location='./cache', verbose=0)


def query(query_string):
    # with CursorManager(credentials) as cursor:
    with connect.connect(credentials) as conn:
        df = pd.read_sql(query_string, conn)
    to_string = ['group', 'dataset', 'shape', 'residual_mode']
    df = df.apply(lambda x: id_to_string(x) if x.name in to_string else x)
    return df


@memory.cache
def cached_query(query_string):
    return query(query_string)


def clear_cache():
    memory.clear()


options = {
    'dataset': ['529_pollen',
                'sleep',
                #         'adult',
                '537_houses',
                #         'nursery',
                '201_pol',
                'mnist',
                'connect_4',
                'wine_quality_white'],
    'agg': ['avg', 'min', 'max'],
    'shape': [
        'rectangle', 'trapezoid', 'exponential',
        'wide_first_2x', 'wide_first_4x', 'wide_first'],
    'loss': ['train_loss', 'train_hinge', 'train_accuracy',
             'test_loss', 'test_hinge', 'test_accuracy',
             'train_squared_hinge', 'train_cosine_similarity',
             'test_squared_hinge', 'train_mean_squared_error',
             'train_mean_absolute_error', 'test_cosine_similarity',
             'test_mean_squared_error', 'train_root_mean_squared_error',
             'test_mean_absolute_error',
             'train_kullback_leibler_divergence',
             'test_root_mean_squared_error',
             'train_mean_squared_logarithmic_error',
             'test_kullback_leibler_divergence',
             'test_mean_squared_logarithmic_error'],
    'residual_mode': ['none', 'full'],
    'group_select': ['min', 'max'],
    'depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
    'size': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
             32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
             8388608, 16777216, 33554432],
    'color_range': (1.05, 2.5, .05),
    'epoch_axis': ['epoch', 'log_effort', 'effort'],
    'learning_rate': [0.001, 0.0001],
    'label_noise': [0.0, 0.05, 0.10, 0.15, 0.20],
    'group': ['fixed_3k_1', 'fixed_3k_0', 'fixed_01', 'exp00', 'exp01']
}

pio.templates['dmp_template'] = go.layout.Template(
    layout=go.Layout(
        colorscale={
            'diverging': 'Temps',
            'sequential': 'Viridis_r',
            'sequentialminus': 'Viridis_r',
            #             'sequential': 'Agsunset_r',
            #             'sequentialminus':'Agsunset_r',
        }
    )
)
pio.templates.default = 'dmp_template'


# print(string_to_id_map)

def setup_value(df, loss, color_range):
    z_label = loss
    #     df['value'] = -np.log(np.minimum(df['value'], np.min(df['value'])*4))
#     df['value'] = -np.log(df['value'] / np.min(df['value']))
#     df['value'] = -np.log(df['value'] / np.min(df['value']))
#     df['value'] = -(df['value'] / np.min(df['value']))

    minimizing = True
    if 'accuracy' in loss:
        df['value'] = 1 - df['value']
        df['value'] = np.minimum(
            np.min(df['value']) * color_range, df['value'])
        df['value'] = np.log(df['value'])/np.log(10)
        z_label = f'log(1-{loss})'
    elif 'loss' in loss:
        #         df['value'] = -np.exp(1 - df['value']/np.min(df['value']))
        df['value'] = np.minimum(
            np.min(df['value']) * color_range, df['value'])
#         df['value'] = -np.log(df['value'])/np.log(10)
#         z_label = f'-log({loss})'

        df['value'] = df['value'] / np.abs(np.min(df['value']))
        z_label = f'loss / abs(min(loss))'
#         df['value'] = -np.log(df['value'] / np.min(df['value']))/np.log(10)
#         df['value'] = -df['value'] / np.min(df['value'])
    elif 'error' in loss:
        df['value'] = df['value'] / np.min(df['value'])
        df['value'] = np.minimum(color_range, df['value'])
        z_label = f'error / min(error)'

    if minimizing:
        best = np.nanmin(df['value'])
    else:
        best = np.nanmax(df['value'])

    return z_label, minimizing, best


def compute_effort(df):
    df['effort'] = (df['epoch'] * df['size'].astype('float')).astype('float')
    df['log_effort'] = np.log(df['effort']) / np.log(10)
    df['relative_effort'] = df['effort'] / np.min(df['effort'])
    df['relative_log_effort'] = np.log(df['relative_effort']) / np.log(10)


def get_values_for_categorical_keys(df, partition_keys):
    partitions = []
    for partition_key in partition_keys:
        partition_values = sorted(df[partition_key].unique())
        partitions.append(
            (partition_key,
             {key: index for index, key in enumerate(partition_values)},
             partition_values
             ))
    return tuple(partitions)


def partitioned_interpolation(df, partition_keys, interpolation_key, value_key, resolution):
    partitions = get_values_for_categorical_keys(df, partition_keys)

    def make_partition_accumulator(i):
        index = partitions[i][1]
        return [make_partition_accumulator(i + 1) if i < len(partitions) - 1 else ([], [])
                for p in range(len(index))]
    acc = make_partition_accumulator(0)

    for _, row in df.iterrows():
        a = acc
        for partition_key, index, _ in partitions:
            a = a[index[row[partition_key]]]
        a[0].append(row[interpolation_key])
        a[1].append(row[value_key])

    interpolation_series = df[interpolation_key]
    interpolation_index = np.linspace(
        np.min(interpolation_series), np.max(interpolation_series), resolution)
    partition_indexes = [np.linspace(0, len(p), len(p)) for p in partitions]

    def do_interpolation(a):
        if type(a) is list:
            return [do_interpolation(p) for p in a]
        func = scipy.interpolate.interp1d(
            a[0], a[1], kind='linear', bounds_error=False, fill_value=np.NaN)
        return func(interpolation_index)
    interpolated = np.array(do_interpolation(acc))
    print(f'interpolated {interpolated.shape}')
    return partitions, interpolation_index, interpolated


def make_2d_heatmap_viz(df, group, dataset, shape, loss, agg, residual_mode, viz, color_range):
    z_label, minimizing, best = setup_value(df, loss, color_range)

    if viz == 'imshow':
        img = df.pivot_table(columns='epoch', index='size', values='value')
        fig = px.imshow(img)
        fig.update_yaxes(type='category')
    elif viz == 'scatter':
        df = df.sort_values(['epoch', 'size'], ascending=[True, False])
        df['size'] = df['size'].astype('str')
        df['epoch'] = df['epoch'].astype('str')
        fig = px.scatter(df, x='epoch', y='size', size='count', color='value')
    elif viz == 'effort':
        compute_effort(df)
        key = 'log_effort'
        x_res = 4000
        partitions, x_index, interpolated = partitioned_interpolation(
            df, ['size'], key, 'value', x_res)
        fig = px.imshow(interpolated, aspect='auto', zmin=np.min(interpolated), zmax=np.max(interpolated),
                        x=x_index,
                        y=[str(b) for b in partitions[0][1]],
                        labels=dict(x='log(Effort)', y='# Parameters', color=z_label),)
        fig.update_yaxes(type='category')
    else:
        return None
    fig.update_layout(
        title=f'{z_label} using {loss} for {dataset}, {shape}, residual {residual_mode}')
    return fig


def saturation_opacity_3d_plot(
    df=None,
    statistic='test_loss_avg',
    dataset=None,
    shape=None,
    viz='imshow',
    surface_within_min=2.0,
    surface_within_max=None,
    xvar='depth',
    yvar='size',
    zvar='log_epoch',
):

    if viz == 'scatter':

        fig = px.scatter_3d(df,
                            x=xvar,
                            y=yvar,
                            z=zvar,
                            color=statistic,
                            log_y=True,
                            opacity=0.25)

        return fig

    elif viz == 'imshow':

        dimensions = [xvar, yvar, zvar]

        print(dimensions)
        df = df.copy()

        df[dimensions[0]] = df[dimensions[0]].astype(int)
        df[dimensions[1]] = df[dimensions[1]].astype(int)
        df[dimensions[2]] = df[dimensions[2]].astype(int)

        x_labels, y_labels, z_labels = [
            sorted(df[dim].unique()) for dim in dimensions]
        X, Y, Z = np.mgrid[0:len(x_labels), 0:len(y_labels), 0:len(z_labels)]
        values = np.empty((X+Y+Z).shape)
        values[:] = np.NaN

        x_idx = {a: b for b, a in enumerate(x_labels)}
        y_idx = {a: b for b, a in enumerate(y_labels)}
        z_idx = {a: b for b, a in enumerate(z_labels)}

        for _, row in df.iterrows():
            values[x_idx[row[dimensions[0]]], y_idx[row[dimensions[1]]],
                   z_idx[row[dimensions[2]]]] = row[statistic]

        minimum = values.min()
        maximum = values.max()

        isomax = maximum
        if surface_within_min is not None:
            isomax = minimum * surface_within_min

        isomin = minimum
        if surface_within_max is not None:
            isomin = maximum * surface_within_max

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomax=isomax,
            isomin=isomin,
            opacity=0.3,
            surface_count=30
        ))

        fig.update_layout(scene=dict(
            xaxis=dict(ticktext=x_labels,
                       tickvals=list(range(0, len(x_labels))),
                       title=dimensions[0]),

            yaxis=dict(ticktext=y_labels,
                       tickvals=list(range(0, len(y_labels))),
                       title=dimensions[1]),

            zaxis=dict(ticktext=z_labels,
                       tickvals=list(range(0, len(z_labels))),
                       title=dimensions[2]),


        ),
            width=700, height=700,
            title=f'{statistic} for {dataset}, {shape}')

        return fig


def heatmap_3d_plot(
        df=None,
        statistic='test_loss_avg',
        dataset=None,
        shape=None,
        viz='imshow'):

    if viz == 'scatter':

        fig = px.scatter_3d(df,
                            x='depth',
                            y='size',
                            z='epoch',
                            color=statistic,
                            log_y=True,
                            opacity=0.25)

        return fig

    elif viz == 'imshow':

        dimensions = ['size', 'depth', 'epoch']

        df[dimensions[0]] = df[dimensions[0]].astype(int)
        df[dimensions[1]] = df[dimensions[1]].astype(int)
        df[dimensions[2]] = df[dimensions[2]].astype(int)

        x_labels, y_labels, z_labels = [
            sorted(df[dim].unique()) for dim in dimensions]
        X, Y, Z = np.mgrid[0:len(x_labels), 0:len(y_labels), 0:len(z_labels)]
        values = np.empty((X+Y+Z).shape)
        values[:] = np.NaN

        x_idx = {a: b for b, a in enumerate(x_labels)}
        y_idx = {a: b for b, a in enumerate(y_labels)}
        z_idx = {a: b for b, a in enumerate(z_labels)}

        for _, row in df.iterrows():
            values[x_idx[row[dimensions[0]]], y_idx[row[dimensions[1]]],
                   z_idx[row[dimensions[2]]]] = row[statistic]

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=30
        ))

        fig.update_layout(scene=dict(
            xaxis=dict(ticktext=x_labels,
                       tickvals=list(range(0, len(x_labels))),
                       title=dimensions[0]),

            yaxis=dict(ticktext=y_labels,
                       tickvals=list(range(0, len(y_labels))),
                       title=dimensions[1]),

            zaxis=dict(ticktext=z_labels,
                       tickvals=list(range(0, len(z_labels))),
                       title=dimensions[2]),


        ),
            width=700, height=700,
            title=f'{statistic} for {dataset}, {shape}')

        return fig


def dump_args(func):
    '''
    This function is from StackOverflow https://stackoverflow.com/a/6278457

    Decorator to print function call details.

    This includes parameters names and effective values.
    '''

    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ', '.join(
            map('{0[0]} = {0[1]!r}'.format, func_args.items()))
        print(f'{func.__module__}.{func.__qualname__} ( {func_args_str} )')
        return func(*args, **kwargs)

    return wrapper


# Old function, contains effort contour lines code.
# def fixed_3k_1_gridded_heatmap(label_noise='0.0', learning_rate='0.0001', depth=None, agg='avg', print_query=False):

#     group = 'fixed_3k_1'

#     if (agg=='cov'):
#         test_str = 'stddev(a.val)/avg(a.val)'
#     else:
#         test_str = f'{agg}(a.val)'

#     query_string = f'''
#     select size, {test_str} as value, count(a.val) as count, a.epoch, dataset, shape, residual_mode
#     from
#         materialized_experiments_3_base base,
#         materialized_experiments_3_test_loss history,
#         unnest(history.test_loss) WITH ORDINALITY as a(val, epoch)
#     WHERE
#         base.id = history.id and
#         'group' = {string_to_id(group)} and
#         {f'depth={depth} and' if depth else ''}
#         learning_rate = {learning_rate}::real and
#         label_noise = {label_noise}::real
#     GROUP BY size, epoch, dataset, shape, residual_mode;
#     '''

#     if print_query:
#         print(query_string)
#         return

#     df = cached_query(query_string)

#     df['log_epoch'] = np.log10((df['epoch']))

#     df['residual_mode'].replace({'none':'', 'full':'full residual\n'}, inplace=True)
#     df['shape'] = df['residual_mode'] + df['shape']

#     df['log_value'] = np.log(df['value'])

#     x_axes = np.linspace(df['log_epoch'].min(), df['log_epoch'].max(), 500)

#     ROWS = 7
#     COLS = 6

#     MIN_BUDGET_EXP = 5
#     MAX_BUDGET_EXP = 24
#     IMAGE_ROWS = MAX_BUDGET_EXP - MIN_BUDGET_EXP + 1

#     fig, axs = plt.subplots(ROWS,COLS, figsize=(11,8), dpi=300, sharex=True)

#     i = 0
#     for idx, gdf in df.groupby(['dataset', 'shape']):

#         #### Plot loss heatmap in each cell

#         img = np.ndarray(shape=(IMAGE_ROWS,500))
#         v_range = gdf['log_value'].max() - gdf['log_value'].min()

#         for row_idx, row_df in gdf.groupby('size'):

#             color_scale = 0.8

#             intp = scipy.interpolate.interp1d(row_df['log_epoch'],
#                                             row_df['log_value'],
#                                             kind='linear',
#                                             bounds_error=False)

#             row_idx = int(np.log2(row_idx)-5) # sizes start at 2^5, and we want this row at the origin

#             vals = intp(x_axes)
#             vals = np.minimum(gdf['log_value'].min()+(v_range*color_scale), vals)
#             img[row_idx, :] = vals

#         ax = axs[i//COLS, i%COLS]
#         pcm = ax.imshow(img, aspect='auto', cmap='viridis_r')#, vmin=value_min, vmax=value_max)

#         #### Effort contour lines

#         X,Y = np.meshgrid(np.arange(0,500),np.arange(0, IMAGE_ROWS))
#         Z = np.power(10, x_axes[X])*np.power(2, Y+5)
#         clevels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 20)
#         ax.contour(Z,  levels=clevels[1::2], colors='white', linestyles='dashed', alpha=0.25)
#         ax.contour(Z,  levels=clevels[::2], colors='black', linestyles='dashed', alpha=0.25)

#         #### Axes style and labels

#         if i%COLS == 0:
#             ax.set_ylabel(idx[0]+'\n\nBudget')
#             ticks = [0, 10, 20]
#             ax.set_yticks(ticks, minor=False)
#             labels = ['$2^{'+str(x+5)+'}$' for x in ticks]
#             ax.set_yticklabels(labels, fontdict=None, minor=False)
#         else:
#             ax.yaxis.set_visible(False)

#         if i//COLS != 6:
#             ax.xaxis.set_visible(False)

#         if i//COLS == 0:
#             ax.set_title(idx[1])

#         if i//COLS == 6:
#             labels = [0,1,2,3]
#             ticks = [np.abs(x_axes-i).argmin() for i in [0,1,2,3]]
#             ax.set_xticks(ticks, minor=False)
#             labels = ['$10^{'+str(x)+'}$' for x in labels]
#             ax.set_xticklabels(labels, fontdict=None, minor=False)
#             ax.set_xlabel('Epoch')

#         i = i + 1

#     #fig.colorbar(pcm, ax=axs, location='right', shrink=0.6)
#     plt.suptitle(f'Average Log Validation Loss [label_noise={label_noise}, learning_rate={learning_rate}, depth={depth if depth else 'Any'}]')
#     #plt.tight_layout()

def load_parameter_map():
    with CursorManager(credentials) as cursor:
        return PostgresParameterMap(cursor)


parameter_map = load_parameter_map()

options_summary = {
    'statistic': [
        'num',
        'test_loss_num_finite',
        'test_loss_avg',
        'test_loss_stddev',
        'test_loss_min',
        'test_loss_max',
        'train_loss_avg'
    ],
    'shape': [
        'rectangle',
        'trapezoid',
        'exponential',
        'wide_first_2x',
        'wide_first_4x',
        'wide_first_16x'],
    'dataset': [
        '529_pollen',
        'sleep',
        '537_houses',
        '201_pol',
        'mnist',
        'connect_4',
        'wine_quality_white']
}


@memory.cache
def get_summary_records(
        with_parameters=None,
        return_values=None,
        print_query=True,
        kinds=None):

    if with_parameters is None:
        with_parameters = []

    if return_values is None:
        return_values = [
            ('num', 'float'),
            ('test_loss_num_finite', 'float'),
            ('test_loss_avg', 'float'),
            ('test_loss_stddev', 'float'),
            ('test_loss_min', 'float'),
            ('test_loss_max', 'float'),
            ('test_loss_percentile', None),
            ('train_loss_avg', 'float'),
            ('train_loss_percentile', None),
        ]
    if kinds is None:
        def infer_type(k):
            types_found = set()
            for v, i in parameter_map.get_all_parameters_for_kind(k):
                types_found.add(type(v))
            if str in types_found:
                return str
            if float in types_found:
                return np.float32
            if int in types_found:
                return 'Int64'
            if bool in types_found:
                return 'boolean'
            return None

        kinds = [(k, infer_type(k)) for k in parameter_map.get_all_kinds()]

    param_map = parameter_map.to_parameter_ids(with_parameters.items())

    query_string = f'''
    SELECT
        experiment_id,
        experiment_parameters,
        num_runs,
        {', '.join([s[0] for s in return_values])}
    FROM
        experiment_summary_
    WHERE
        experiment_parameters @> (array[{ param_map }])::smallint[];
    '''

    if print_query:
        print(query_string)

    print('Querying...')
    df = cached_query(query_string).copy()
    print('Data recieved.')

    print('Interpeting parameters...')

    def make_parameter_map(parameter_id_list):
        return {kind: value
             for kind, value in
             parameter_map.parameter_from_id(parameter_id_list)}
        # return tuple((m.get(kind[0], None) for kind in kinds))


    parameter_maps = df['experiment_parameters'].map(make_parameter_map)

    # def make_parameter_map(parameter_id_list):
    #     m = {kind: value
    #          for kind, value in
    #          parameter_map.parameter_from_id(parameter_id_list)}
    #     return tuple((m.get(kind[0], None) for kind in kinds))

    # parameters = list(
    #     zip(*df['experiment_parameters'].map(make_parameter_map)))

    print('Converting parameters...')
    for i, (kind, typename) in enumerate(kinds):
        df[kind] = parameter_maps.map(lambda m : m.get(kind, None))
        if typename is not None:
            df[kind] = df[kind].astype(typename)

    df.drop(columns=['experiment_parameters'], inplace=True)

    print('Expanding epochs...')
    df['epoch'] = df['num'].apply(lambda x: list(range(1, 1 + len(x))))
    return_values.append(('epoch', 'int'))

    df = df.explode([s[0] for s in return_values])
    for s, dtype in return_values:
        if dtype is not None:
            df[s] = df[s].astype(dtype)
    print('Done.')
    return df


@memory.cache
def fixed_3k_1_gridded_heatmap_summary_get_data(
        label_noise=0.0,
        learning_rate=0.0001,
        depth=3,
        batch='fixed_3k_1',
        statistics=None,
        kinds=None,
        dataset=None,
        size=None,
        shape=None,
        print_query=True):

    if statistics is None:
        statistics = [
            ('num', 'float'),
            ('test_loss_num_finite', 'float'),
            ('test_loss_avg', 'float'),
            ('test_loss_stddev', 'float'),
            ('test_loss_min', 'float'),
            ('test_loss_max', 'float'),
            # ('test_loss_percentile', None),
            ('test_loss_median', None),
            ('train_loss_avg', 'float'),
            # ('train_loss_percentile', None),
            ('train_loss_median', None),
            ('train_mean_squared_error_median', None),
            ('test_mean_squared_error_median', None),
            ('train_kullback_leibler_divergence_median', None),
            ('test_kullback_leibler_divergence_median', None),
        ]

    if kinds is None:
        kinds = [
            ('batch', 'str'),
            ('dataset', 'str'),
            ('learning_rate', 'float'),
            ('label_noise', 'float'),
            ('shape', 'str'),
            ('depth', 'int'),
            ('size', 'int'),
        ]

    params_premap = [x for x in [
        ('batch', batch),
        ('depth', depth),
        ('learning_rate', learning_rate),
        ('label_noise', label_noise),
        ('dataset', dataset),
        ('size', size),
        ('shape', shape)
    ] if x[1] is not None]

    # print(params_premap)

    param_map = parameter_map.to_parameter_ids(params_premap)

    query_string = f'''
    SELECT
        experiment_id,
        num_free_parameters,
        widths,
        relative_size_error,
        experiment_parameters,
        num_runs,
        {', '.join([s[0] for s in statistics])}
    FROM
        experiment_summary_
    WHERE
        experiment_parameters @> (array[{ param_map }])::smallint[];
    '''

    if print_query:
        print(query_string)

    print('Querying...')
    df = cached_query(query_string).copy()
    print('Data recieved.')

    print('Interpeting parameters...')

    def make_parameter_map(parameter_id_list):
        m = {kind: value for kind,
             value in parameter_map.parameter_from_id(parameter_id_list)}
        return tuple((m.get(kind[0], None) for kind in kinds))

    parameters = list(
        zip(*df['experiment_parameters'].map(make_parameter_map)))

    print('Converting parameters...')
    for i, (kind, typename) in enumerate(kinds):
        df[kind] = parameters[i]
        df[kind] = df[kind].astype(typename)

    df.drop(columns=['experiment_parameters'], inplace=True)

    print('Expanding epochs...')
    df['epoch'] = df['num'].apply(lambda x: list(range(1, 1 + len(x))))
    statistics.append(('epoch', 'int'))

    df = df.explode([s[0] for s in statistics])
    for s, dtype in statistics:
        if dtype is not None:
            df[s] = df[s].astype(dtype)
    print('Done.')
    return df


def prepare_heatmap_axis_ticks(
    variable_name,
    variable_values,
    axis,
    num_ticks,
    include_label=True,
    extent_mode=False,
    scaler = None
):
    if scaler is None:
        scaler = lambda x : x
        
    labels = None
    ticks = None
    label = variable_name

    def make_ticks(tick_values):
        if extent_mode:
            return tick_values
        else:
            return [np.abs(variable_values-i).argmin() for i in tick_values]

    if variable_name == 'log_epoch':
        tick_values = [0, 1, 2, 3]
        ticks = make_ticks(tick_values)
        labels = ['$10^{'+str(x)+'}$' for x in tick_values]
        label = 'Epoch'
    elif variable_name == 'epoch':
        tick_values = [0, 1e3, 2e3, 3e3]
        ticks = make_ticks(tick_values)
        labels = [f'${v}$' for v in tick_values]
        label = 'Epoch'
    elif variable_name == 'depth':
        tick_values = [5, 10, 15]
        ticks = make_ticks(tick_values)
        labels = [str(x) for x in tick_values]
        label = 'Depth'
    elif variable_name == 'effort':
        frac = 100
        ticks = [frac * i for i in range(num_ticks)]
        labels = ['$10^{'+str(x)+'}$' for x in ticks]
        label = 'Effort'
    elif variable_name == 'log_effort':
        frac = 100
        v = np.array(variable_values)
        order = np.floor(np.log10(1e-100 + np.max(np.abs(v)))) - 1
        scale = 10 ** order
        min_tick = scale * np.ceil(np.min(v) / scale)
        max_tick = scale * np.floor(np.max(v) / scale)
        num_ticks = int(np.floor((max_tick - min_tick) / scale))
        tick_values = [min_tick + i * scale for i in range(num_ticks)]
        tick_values = [int(v) if v == round(v) else v for v in tick_values]
        ticks = make_ticks(tick_values)
        # ticks = [frac * i for i in range(num_ticks)]
        labels = ['$10^{'+str(x)+'}$' for x in tick_values]
        label = 'Effort'
    elif variable_name == 'test_loss':
        frac = 100
        v = np.array(variable_values)
        order = np.floor(np.log10(1e-100 + np.max(np.abs(v)))) - 1
        scale = 10 ** order
        min_tick = scale * np.ceil(np.min(v) / scale)
        max_tick = scale * np.floor(np.max(v) / scale)

        nt = 0
        while True:
            nt = int(np.floor((max_tick - min_tick) / scale))
            if nt <= num_ticks and nt > 2:
                break
            scale *= 2

        tick_values = [min_tick + i * scale for i in range(nt)]
        tick_values = [int(v) if v == round(v) else v for v in tick_values]
        ticks = make_ticks(tick_values)
        labels = ['$10^{'+format(x, ".2f")+'}$' for x in tick_values]
        label = 'Test Loss'
    elif variable_name == 'log_test_loss':
        frac = 100
        v = np.array(variable_values)
        order = np.floor(np.log10(1e-100 + np.max(np.abs(v)))) - 1
        scale = 10 ** order
        min_tick = scale * np.ceil(np.min(v) / scale)
        max_tick = scale * np.floor(np.max(v) / scale)
        nt = 0
        while True:
            nt = int(np.floor((max_tick - min_tick) / scale))
            if nt <= num_ticks or nt <= 2:
                break
            scale *= 2

        tick_values = [min_tick + i * scale for i in range(nt)]
        tick_values = [int(v) if v == round(v) else v for v in tick_values]
        ticks = make_ticks(tick_values)
        labels = ['$10^{'+format(x, '.2f')+'}$' for x in tick_values]
        label = 'Test Loss'
    elif variable_name == 'log_size' or variable_name == 'size':
        work_vals = variable_values
        if variable_name == 'size':
            work_vals = np.log(work_vals) / np.log(2)

        # label = f'{variable_group}\n\n{variable_name}'
        label = '# Parameters'
        # ax.set_ylabel(f'{y_value}\n\n{y_axis}')

        tick_values = work_vals
        if len(work_vals) > num_ticks:
            frac = (len(work_vals)-1) / (num_ticks-1)
            tick_values = [work_vals[int(i * frac)]
                           for i in range(num_ticks)]

        labels = ['$2^{' +
                  str(int(np.round(v)))
                  + '}$'
                  for v in tick_values]

        if variable_name == 'size':
            tick_values = np.power(2, tick_values)
        ticks = make_ticks(tick_values)

        # ax.set_yticks(ticks, minor=False)
        # ax.set_yticklabels(labels, fontdict=None, minor=False)
    elif variable_name == 'num_free_parameters':
        label = '# Free Parameters'

        ticks = variable_values
        if len(variable_values) > num_ticks:
            frac = len(variable_values) / (num_ticks-1)
            ticks = [min(len(variable_values)-1, int(i * frac))
                     for i in range(num_ticks)]
        labels = ['$2^{' +
                  str(int(np.round(np.log(variable_values[t]) / np.log(2))))
                  + '}$'
                  for t in ticks]
    else:
        assert False

    axis.set_ticks(ticks, labels=labels, minor=False)
    if include_label:
        axis.set_label_text(label)
    axis.set_visible(True)


def fixed_3k_1_gridded_heatmap_v2_plot(df, title,
                                       color_type='scalar',
                                       color_bar=True,
                                       y_variable='log_epoch',
                                       x_variable='size',
                                       y_resolution=1000,
                                       group_names=('shape', 'dataset'),
                                       hue=127/255.,
                                       num_x_ticks=4,
                                       num_y_ticks=4,
                                       contour_color=[[1.0, .25, .25]],
                                       contour_alpha=.8,
                                       contour_linewidth=1.0,
                                       include_contour=True,
                                       parameter_dict=None,
                                       contour_data=None,
                                       contour_levels=[0.70],
                                       cmap='viridis',
                                       vmin=None,
                                       vmax=None,
                                       y_scaler=None,
                                       y_min=1,
                                       y_max=3000,
                                       extent_mode = False,
                                       ):

    if y_scaler is None:
        y_scaler = lambda x: np.log10(x)

    group_series = [df[name] for name in group_names]
    group_values = [np.sort(frame.unique()) for frame in group_series]
    
    contours = dict()

    def interpolate_col(x, y):
        return scipy.interpolate.interp1d(
            x,
            y,
            kind='linear',
            bounds_error=False
        )(y_values)
    
    def objective(x, size_offset, epoch_offset, exponent, constant):
        return (constant/((x - size_offset)**exponent)) + epoch_offset
    
    subplot_sizes = (len(group_values[1]), len(group_values[0]))
    fig, axes = plt.subplots(*subplot_sizes, figsize=(7.25, 14), dpi=400)

    

    axs = []
    for col_idx, col_group_value in enumerate(group_values[0]):
        col_df = df[group_series[0] == col_group_value]
        contours[col_group_value] = dict()
        for row_idx, row_group_value in enumerate(group_values[1]):
            cell_df = col_df[col_df[group_names[1]] == row_group_value]

            if len(cell_df)==0:
                img = np.empty(shape=(y_resolution, len(x_values)))
                img[:] = np.nan
                ax = axes[row_idx, col_idx]
                axs.append(ax)
                ax_imshow = ax.imshow(
                    img,
                    cmap=cmap,
                    origin='lower',
                    aspect='auto',
                    interpolation='nearest',
                    **imshow_kwargs)
                    
                ax.set_xticks([])
                ax.set_xticks([],minor=True)
                ax.set_yticks([])
                ax.set_yticks([],minor=True)
                continue
#             y_values = np.linspace(cell_df['y'].min(), cell_df['y'].max(), y_resolution)
            y_values = np.linspace(y_scaler(y_min), y_scaler(y_max), y_resolution, endpoint=True)

            x_values = np.sort(cell_df[x_variable].unique())
#             if x_variable == 'num_free_parameters': # Add missing sizes
#                 x_values = np.concatenate(([2**x for x in range(5,26) if x < np.log2(np.min(x_values))],
#                                            x_values,
#                                            [2**x for x in range(5,26) if x > np.log2(np.max(x_values))]))
            
            extent = None
            if extent_mode:
                dx = (x_values[1]-x_values[0])/2.
                dy = (y_values[1]-y_values[0])/2.
                extent = [x_values[0]-dx, x_values[-1]+dx, y_values[0]-dy, y_values[-1]+dy]
                # extent = [np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)]


            if color_type == 'scalar':
                img = np.ndarray(shape=(y_resolution, len(x_values)))
#                 imshow_kwargs = {'vmin': 0, 'vmax': 1}
                imshow_kwargs = {}
            if color_type == 'rgb' or color_type == 'saturation':
                img = np.ndarray(shape=(y_resolution, len(x_values), 3))
                imshow_kwargs = {}

            for size_idx, size in enumerate(x_values):

                img_row_df = cell_df[cell_df[x_variable] == size]

                if len(img_row_df) == 0:
                    img[:, size_idx] = None
                    continue

                if color_type == 'scalar':
                    img[:, size_idx] = interpolate_col(
                        img_row_df['y'], img_row_df['color']).flatten()
                elif color_type == 'rgb':
                    img[:, size_idx, 0] = interpolate_col(
                        img_row_df['y'], img_row_df['color_r'])
                    img[:, size_idx, 1] = interpolate_col(
                        img_row_df['y'], img_row_df['color_g'])
                    img[:, size_idx, 2] = interpolate_col(
                        img_row_df['y'], img_row_df['color_b'])
                elif color_type == 'saturation':
                    saturation = interpolate_col(
                        img_row_df['y'], img_row_df['color_s'])
                    value = interpolate_col(
                        img_row_df['y'], img_row_df['color_v'])
                    img[:, size_idx, :] = np.array(list(
                        map(lambda t: colorsys.hsv_to_rgb(hue, t[0], t[1]),
                            zip(saturation, value))))

            ax = axes[row_idx, col_idx]
            axs.append(ax)

#             print (img.shape)
            
            ax_imshow = ax.imshow(
                img,
                cmap=cmap,
                vmax=vmax,
                vmin=vmin,
                origin='lower',
                aspect='auto',
                interpolation='nearest',
                extent = extent,
                **imshow_kwargs)
            
            if include_contour:
                X,Y = np.meshgrid(np.arange(len(x_values)), np.arange(0, len(y_values)))
                contours[col_group_value][row_group_value] = ax.contour(X, Y, img, levels=contour_levels, colors='r', 
                                                               linestyles='dashed',alpha=contour_alpha,
                                                               linewidths=contour_linewidth,)
            if parameter_dict != None:
                row_parameters = parameter_dict[row_group_value][col_group_value]
                ax.plot(np.log2(x_values)-5, [np.log10(objective(x,row_parameters[0][0], 
                                               row_parameters[0][1], 
                                               row_parameters[0][2], 
                                               row_parameters[0][3]))/0.003477265995424853 for x in x_values],
                        c='orange',linestyle='dashed')
                try:
                    ax.plot(np.log2(x_values)-5, [np.log10(row_parameters[3](x))/0.003477265995424853 for x in x_values],
                            c='white',linestyle='dashed')
                except:
                    pass
                
                           
#                 if not row_parameters['upper_index'].isna().values[0]:
#                     ax.plot(np.log2(x_values)-5, [np.log10(objective(x,row_parameters['upper_size_offset'], 
#                                                row_parameters['upper_epoch_offset'], 
#                                                row_parameters['upper_exponents'], 
#                                                row_parameters['upper_constants']))/0.003477265995424853 for x in x_values],
#                             c='orange',linestyle='dashed')
                ax.set_ylim(0,len(y_values))
                ax.set_xlim(0,len(x_values))
            
#             if y_variable == 'log_epoch':
#                 X, Y = np.meshgrid(np.arange(0, len(x_values)),
#                                    np.arange(0, len(y_values)))
#                 Z = np.power(10, y_values[Y])*np.power(2, X+5)
#                 clevels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 6)
#                 ax.contour(
#                     Z,
#                     levels=clevels,
#                     colors=contour_color,
#                     linestyles='dashed',
#                     alpha=contour_alpha,
#                     linewidths=contour_linewidth,
#                 )
#             elif y_variable == 'epoch':
#                 X, Y = np.meshgrid(np.arange(0, len(x_values)),
#                                    np.arange(0, len(y_values)))
#                 Z = y_values[Y]*np.power(2, X+5)
#                 clevels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 6)
#                 ax.contour(
#                     Z,
#                     levels=clevels,
#                     colors=contour_color,
#                     linestyles='dashed',
#                     alpha=contour_alpha,
#                     linewidths=contour_linewidth,
#                 )

        

            if contour_data != None:
                Z = np.ndarray(shape=(y_resolution, len(x_values)))
                for i,x in enumerate(x_values):
                    img_row_df = cell_df[cell_df[x_variable] == x]
                    Z[:, i] = interpolate_col(img_row_df['y'],
                                              img_row_df[contour_data],
                                             ).flatten()
                clevels = np.linspace(np.min(Z), np.max(Z), 6)
                ax.contour(
                    Z,
                    levels=clevels,
                    colors=contour_color,
                    linestyles='dashed',
                    alpha=contour_alpha,
                    linewidths=contour_linewidth,
                )

            ax.xaxis.set_visible(False)
            if row_idx == 0:
                ax.set_title(col_group_value)

            if (row_idx >= len(group_values[1]) - 1):
                prepare_heatmap_axis_ticks(
                    x_variable, x_values, ax.xaxis, num_x_ticks, extent_mode=extent_mode)
                
            if (x_variable == 'num_free_parameters'):
                prepare_heatmap_axis_ticks(
                    x_variable, x_values, ax.xaxis, num_x_ticks, include_label=False, extent_mode=extent_mode)

            ax.yaxis.set_visible(False)
            if col_idx == 0:
                prepare_heatmap_axis_ticks(
                    y_variable, y_values, ax.yaxis, num_y_ticks, scaler = y_scaler, extent_mode=extent_mode)

            if col_idx >= len(group_values[0]) - 1:
                row_axis = fig.add_subplot(
                    *subplot_sizes,
                    (col_idx+1) + (len(group_values[0]) * row_idx),
                    # sharex=ax,
                    # sharey=ax,
                    frameon=False,
                )
                axs.append(row_axis)

                row_axis.xaxis.set_ticks([], [])
                row_axis.yaxis.set_ticks([], [])
                row_y_axis = row_axis.yaxis
                row_y_axis.tick_right()
                # row_y_axis.labelpad = -8
                row_y_axis.set_label_position('right')
                row_y_axis.set_label_text(row_group_value)

    plt.tight_layout()
                
    if color_bar:
        fig.colorbar(
            ax_imshow,
            ax=axs,
            orientation="horizontal",
            pad=0.1)

    plt.suptitle(title,y=1)

    
    if include_contour:
        return fig,contours
    else:
        return fig

    
def objective(x, size_offset, epoch_offset, exponent, constant):
    return (constant/((x - size_offset)**exponent)) + epoch_offset

def loss_objective(x, offset, exponent, constant):
    return (constant/((x)**exponent)) + offset

def linear(x, rate, b):
    return ((rate*x) + b)

def quadratic(x, rate):
    return (rate*(x**2))

def softplus(x): 
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def three_dim_objective(size, epoch, 
                        interpolation_size, interpolation_epoch, interpolation_exponent, interpolation_constant,
                        loss_offset, loss_exponent, loss_constant,
                        underfit_descent, overfit_ascent,descent_function='quadratic'):
    """
    1. Use the size to predict the interpolation threshold epoch using the objective function
    2. Use the size to predict the interpolation threshold loss using the objective function
    3. Calculate the distance to the interpolation threshold epoch
    4. Use the quadratic function to predict the loss based on the distance to the minimum epoch
    """
    # 1.
    min_epoch = (interpolation_constant/((size - interpolation_size)**interpolation_exponent)) + interpolation_epoch
    
    # 2.
    min_loss = (loss_constant/((size)**loss_exponent)) + loss_offset
    
    # 3.
    distance_to_min_epoch =  epoch - min_epoch
    
    # step function to choose slope (either underfitting slope or overfitting slope)
    slope = np.heaviside(distance_to_min_epoch,0)*overfit_ascent + np.heaviside(-distance_to_min_epoch,1)*underfit_descent
    
    # 4.
    if descent_function=='linear':
        loss = slope*np.abs(distance_to_min_epoch) + min_loss
    elif descent_function=='softplus':
        loss = slope*softplus(distance_to_min_epoch) + min_loss
    else:
        loss = slope*(distance_to_min_epoch**2) + min_loss
    
    return loss
    
def three_dim_objective_curve_fit(size_epoch, 
                                  interpolation_size, interpolation_epoch, interpolation_exponent, interpolation_constant,
                                  loss_offset, loss_exponent, loss_constant,
                                  underfit_descent, overfit_ascent):
    """
    Interface function for passing size and epoch as list/tuple
    """
    return three_dim_objective(size_epoch[0],size_epoch[1], interpolation_size, interpolation_epoch, interpolation_exponent, 
                               interpolation_constant,loss_offset, loss_exponent, loss_constant,
                               underfit_descent, overfit_ascent)

def three_dim_objective_curve_fit_linear(size_epoch, 
                                  interpolation_size, interpolation_epoch, interpolation_exponent, interpolation_constant,
                                  loss_offset, loss_exponent, loss_constant,
                                  underfit_descent, overfit_ascent):
    """
    Interface function for passing size and epoch as list/tuple
    """
    return three_dim_objective(size_epoch[0],size_epoch[1], interpolation_size, interpolation_epoch, interpolation_exponent, 
                               interpolation_constant,loss_offset, loss_exponent, loss_constant,
                               underfit_descent, overfit_ascent,descent_function='linear')
    
    
def plot_fit(df,fits_df,dataset,shape,x_var,y_var,
             num_free_parameters=None,threshold=None,ax=None):
    if ax == None:
        fig,ax = plt.subplots(1,1,figsize=(5,5))
    
    if num_free_parameters == None:
        data = df[(df['dataset']==dataset) &
                  (df['shape']==shape)]
        fits = fits_df[(fits_df['dataset']==dataset) &
                  (fits_df['shape']==shape)]
    else:
        data = df[(df['dataset']==dataset) &
                  (df['shape']==shape) &
                  (df['num_free_parameters']==num_free_parameters)]
        fits = fits_df[(fits_df['dataset']==dataset) &
                  (fits_df['shape']==shape) &
                  (fits_df['num_free_parameters']==num_free_parameters)]
    
    if threshold != None:
        fits = fits[fits['percent minimum']==threshold]
    
    x = data[x_var]
    y = data[y_var]
    
    ax.loglog(x,y)
    
    if 'intercept' in fits.columns:
        ax.loglog(x,
                  [10**linear(np.log10(xi),fits['slope'],fits['intercept']) for xi in x])
    else:
        ax.loglog(x,
                  [10**objective(np.log10(xi),fits['size_offset'],fits['epoch_offset'],
                                 fits['exponent'],fits['constant']) 
                   for xi in x])

    return ax


