import pytest
import json
import sys
import random

import tensorflow as tf
import numpy.testing as npt

from dmp.data.logging import read_file
from dmp.experiment.aspect_test import aspect_test

"""
SQL query to grab results for the test

SELECT * FROM public.log where
doc -> 'config' ->> 'dataset' = 'wine_quality_white' and
doc -> 'config' ->> 'depth' = '4' and
doc -> 'config' ->> 'topology' = 'rectangle' and
doc -> 'config' ->> 'budget' = '1024'
LIMIT 100
"""

# wine_quality_white, 529 pollen
# change aspect test to have a main function
# once for each architecture. Do it with a depth thats larger than 2
# compare results, setting a seed 
#   check validation loss at the end using float comparison
#   iterations to converge
#       with the seed I don't know how effective it will be - if you can track em down.
#       or run a few times and check to see if the average or SD is similar

def test_aspect_test_historical():
    """
    Read a json file and test loss against re-run using parameters from conf
    Here, the decimal is set super low because we no longer have the original random seed
    """
    data = read_file("tests/data/wine_quality_white__rectangle__1024__4__16105622897956.json")
    data["config"]["residual_mode"] = "none"
    data["config"]["seed"] = 42

    result = aspect_test(data["config"])
    npt.assert_almost_equal(data["loss"], result["loss"], decimal=2)
    npt.assert_almost_equal(data["val_loss"], result["val_loss"], decimal=1)

    print(data["loss"], result["loss"]) # to see this, use pytest -s
    print(data["val_loss"], result["val_loss"]) # to see this, use pytest -s

def test_fixed_regression():
    """
    Run the test against a matrix of data for which we know the random seed. Therefore, results can be exact.
    """
    pass

if __name__=="__main__":
    test_aspect_test()