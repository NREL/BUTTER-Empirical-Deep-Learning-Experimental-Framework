import pytest

def test_pmlb():

    from dmp.data.pmlb import pmlb_loader

    datasets = pmlb_loader.load_dataset_index()
    dataset, inputs, outputs = pmlb_loader.load_dataset(datasets, '537_houses')

    assert dataset['n_observations']==20640
    assert dataset['n_features']==8


