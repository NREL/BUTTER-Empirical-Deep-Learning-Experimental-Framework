import datetime
from joblib import Memory
import inspect
import scipy

from dmp.data.logging import _get_sql_engine
from dmp.data.logging import get_environment

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

env = get_environment()
print(f"""
DMP Viz Tools
Git Commit Hash: {env["git_hash"]}
Hostname: {env["hostname"]}
Date: {datetime.datetime.now()}
""")

db = _get_sql_engine()
memory = Memory(location="./cache", verbose=0)

def query(query_string):
    with db.connect() as engine:
        df = pd.read_sql(query_string, engine)
        to_string = ["group", "dataset", "topology", "residual_mode"]
        df = df.apply(lambda x: id_to_string(x) if x.name in to_string else x)
        return df

@memory.cache
def cached_query(query_string):
    return query(query_string)

def clear_cache():
    memory.clear()
    
options = {
    "dataset": ['529_pollen',
        'sleep',
#         'adult',
        '537_houses',
#         'nursery',
        '201_pol',
        'mnist',
        'connect_4',
        'wine_quality_white'],
    "agg": ["avg", "min", "max"],
    "topology" : [
        "rectangle", "trapezoid", "exponential",
        "wide_first_2x", "wide_first_4x", "wide_first"],
     "loss": ['loss', 'hinge', 'accuracy',
       'test_loss', 'test_hinge', 'test_accuracy',
       'squared_hinge', 'cosine_similarity',
       'test_squared_hinge', 'mean_squared_error',
       'mean_absolute_error', 'test_cosine_similarity',
       'test_mean_squared_error', 'root_mean_squared_error',
       'test_mean_absolute_error',
       'kullback_leibler_divergence',
       'test_root_mean_squared_error',
       'mean_squared_logarithmic_error',
       'test_kullback_leibler_divergence',
       'test_mean_squared_logarithmic_error'],
    "residual_mode": ["none", "full"],
    "group_select" : ["min", "max"],
    'depth':[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
    'budget': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                8388608, 16777216, 33554432],
    'color_range': (1.05, 2.5, .05),
    'epoch_axis' : ['epoch', 'log_effort', 'effort'],
    'learning_rate': [0.001, 0.0001],
    'label_noise': [0.0, 0.05, 0.10, 0.15, 0.20],
    'group' : ['fixed_3k_1', 'fixed_3k_0', 'fixed_01', 'exp00', 'exp01']   
}

pio.templates["dmp_template"] = go.layout.Template(
    layout=go.Layout(
        colorscale={
            'diverging':'Temps',
            'sequential': 'Viridis_r',
            'sequentialminus':'Viridis_r',
#             'sequential': 'Agsunset_r',
#             'sequentialminus':'Agsunset_r',
        }
    )
)
pio.templates.default = 'dmp_template'


string_map_df = pd.read_sql(
        f'''SELECT id, value from strings''',
        db.execution_options(stream_results=True, postgresql_with_hold=True), coerce_float=False,
        params=())

string_to_id_map = {}
id_to_string_map = {}

values = string_map_df['value'].to_list()
for i, str_id in enumerate(string_map_df['id'].to_list()):
    string_to_id_map[values[i]] = str_id
    id_to_string_map[str_id] = values[i]
    
def string_to_id(s):
    if isinstance(s, str):
        return string_to_id_map[s]
    return [string_to_id(e) for e in s]

def id_to_string(i):
    if isinstance(i, int):
        return id_to_string_map[i]
    return [id_to_string(e) for e in i]

#print(string_to_id_map)

def setup_value(df, loss, color_range):
    z_label = loss
    #     df["value"] = -np.log(np.minimum(df["value"], np.min(df["value"])*4))   
#     df["value"] = -np.log(df['value'] / np.min(df['value']))
#     df["value"] = -np.log(df['value'] / np.min(df['value']))
#     df["value"] = -(df['value'] / np.min(df['value']))

    minimizing = True
    if 'accuracy' in loss:
        df['value'] = 1 - df['value']
        df['value'] = np.minimum(np.min(df['value']) * color_range, df['value'])
        df["value"] = np.log(df['value'])/np.log(10)
        z_label = f'log(1-{loss})'
    elif 'loss' in loss:
#         df["value"] = -np.exp(1 - df['value']/np.min(df['value']))
        df['value'] = np.minimum(np.min(df['value']) * color_range, df['value'])
#         df["value"] = -np.log(df['value'])/np.log(10)
#         z_label = f'-log({loss})'
        
        df["value"] = df['value'] / np.abs(np.min(df['value']))
        z_label = f'loss / abs(min(loss))'
#         df["value"] = -np.log(df['value'] / np.min(df['value']))/np.log(10)
#         df["value"] = -df['value'] / np.min(df['value'])
    elif 'error' in loss:
        df["value"] = df['value'] / np.min(df['value'])
        df['value'] = np.minimum(color_range, df['value'])
        z_label = f'error / min(error)'
    
    if minimizing:
        best = np.nanmin(df['value'])
    else:
        best = np.nanmax(df['value'])
        
    return z_label, minimizing, best

def compute_effort(df):
    df["effort"] = (df["epoch"] * df["budget"].astype("float")).astype("float")
    df["log_effort"] = np.log(df["effort"]) / np.log(10)
    df['relative_effort'] = df['effort'] / np.min(df['effort'])
    df['relative_log_effort'] = np.log(df['relative_effort']) / np.log(10)


def get_values_for_categorical_keys(df, partition_keys):
    partitions = []
    for partition_key in partition_keys:
        partition_values = sorted(df[partition_key].unique())
        partitions.append(
            (partition_key,
            {key : index for index, key in enumerate(partition_values)},
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
    interpolation_index = np.linspace(np.min(interpolation_series), np.max(interpolation_series), resolution)
    [np.linspace(0, len(p), len(p)) for p in partitions]
    
    def do_interpolation(a):
        if type(a) is list:
            return [do_interpolation(p) for p in a]
        func = scipy.interpolate.interp1d(a[0], a[1], kind='linear', bounds_error=False, fill_value=np.NaN)
        return func(interpolation_index)            
    interpolated = np.array(do_interpolation(acc))
    print(f'interpolated {interpolated.shape}')
    return partitions, interpolation_index, interpolated


def make_2d_heatmap_viz(df, group, dataset, topology, loss, agg, residual_mode, viz, color_range):
    z_label, minimizing, best = setup_value(df, loss, color_range)
        
    if viz == "imshow":
        img = df.pivot_table(columns="epoch", index="budget", values="value")
        fig = px.imshow(img)
        fig.update_yaxes(type='category')
    elif viz == "scatter":
        df = df.sort_values(["epoch", "budget"], ascending=[True, False])
        df["budget"] = df["budget"].astype("str")
        df["epoch"] = df["epoch"].astype("str")
        fig = px.scatter(df, x="epoch", y="budget", size="count", color="value")
    elif viz == "effort":
        compute_effort(df)
        key = 'log_effort'
        x_res = 4000
        partitions, x_index, interpolated = partitioned_interpolation(df, ['budget'], key, 'value', x_res)
        fig= px.imshow(interpolated, aspect='auto', zmin=np.min(interpolated), zmax=np.max(interpolated),
            x = x_index,
            y = [str(b) for b in partitions[0][1]],
            labels=dict(x="log(Effort)", y="# Parameters", color=z_label),)
        fig.update_yaxes(type='category')
    else:
        return None
    fig.update_layout(title=f"{z_label} using {loss} for {dataset}, {topology}, residual {residual_mode}")
    return fig



# @interact_manual(**options, viz=["imshow", "scatter"])
# def heatmap_app_3d(groups="('exp00', 'exp01')", dataset="537_houses", topology="wide_first", loss="history_test_mean_squared_error", agg="avg", residual_mode="none", viz="imshow"):
#     query_string = f'''
#     select "config.budget", "config.depth", {agg}(a.val) as value, count(a.val), a.epoch
#     from
#         materialized_experiments_0 t,
#         unnest(t.{loss}) WITH ORDINALITY as a(val, epoch)
#     WHERE
#         "groupname" IN {groups} and
#         "config.dataset"='{dataset}' and
#         "config.topology"='{topology}' and
#         "config.residual_mode"='{residual_mode}'
#     GROUP BY epoch, "config.budget", "config.depth"
#     '''
#     df = cached_query(query_string)#.query("count == 30")
    
#     if viz=="scatter":
        
#         fig = px.scatter_3d(df,
#                     x='config.depth',
#                     y='config.budget',
#                     z='epoch',
#                     color='value',
#                     log_y=True,
#                     opacity=0.25)
#         fig.show()
        
#     elif viz=="imshow":

#         dimensions = ["config.budget", "config.depth", "epoch"]

#         df[dimensions[0]] = df[dimensions[0]].astype(int)
#         df[dimensions[1]] = df[dimensions[1]].astype(int)
#         df[dimensions[2]] = df[dimensions[2]].astype(int)

#         x_labels, y_labels, z_labels = [sorted(df[dim].unique()) for dim in dimensions]
#         X, Y, Z = np.mgrid[0:len(x_labels), 0:len(y_labels), 0:len(z_labels)]
#         values = np.empty((X+Y+Z).shape)
#         values[:] = np.NaN

#         x_idx = {a:b for b,a in enumerate(x_labels)}
#         y_idx = {a:b for b,a in enumerate(y_labels)}
#         z_idx = {a:b for b,a in enumerate(z_labels)}

#         for _, row in df.iterrows():
#             values[x_idx[row[dimensions[0]]], y_idx[row[dimensions[1]]], z_idx[row[dimensions[2]]]] = row["value"]


#         fig = go.Figure(data=go.Volume(
#             x=X.flatten(),
#             y=Y.flatten(),
#             z=Z.flatten(),
#             value=values.flatten(),
#             opacity=0.1, # needs to be small to see through all surfaces
#             surface_count=30
#             ))

#         fig.update_layout(scene = dict(
#                             xaxis = dict(ticktext=x_labels,
#                                          tickvals=list(range(0,len(x_labels))),
#                                          title=dimensions[0]),

#                             yaxis = dict(ticktext=y_labels,
#                                          tickvals=list(range(0,len(y_labels))),
#                                          title=dimensions[1]),

#                             zaxis = dict(ticktext=z_labels,
#                                          tickvals=list(range(0,len(z_labels))),
#                                          title=dimensions[2]),


#                             ),
#                           width=700, height=700,
#                          title=f"{loss} for {dataset}, {topology}, residual {residual_mode}")
#         fig.show()

def dump_args(func):
    """
    This function is from StackOverflow https://stackoverflow.com/a/6278457

    Decorator to print function call details.

    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"{func.__module__}.{func.__qualname__} ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper

