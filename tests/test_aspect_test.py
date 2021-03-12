import pytest
import runpy
import json
import sys

import tensorflow as tf
import random

from dmp.data.logging import read_file

"""
SQL query to grab results for the test

SELECT * FROM public.log where
doc -> 'config' ->> 'dataset' = 'wine_quality_white' and
doc -> 'config' ->> 'depth' = '4' and
doc -> 'config' ->> 'topology' = 'rectangle' and
doc -> 'config' ->> 'budget' = '1024'
LIMIT 100
"""

def test_aspect_test():
    data = read_file("tests/data/wine_quality_white__rectangle__1024__4__16105622897956.json")
    
    config = data["config"]
    config["budgets"] = [config["budget"]]
    config["depths"] = [config["depth"]]
    config["topologies"] = [config["topology"]]
    config["log"] = "/tmp"
    
    sys.argv[1] = json.dumps(data["config"])

    tf.random.set_seed(42)

    runpy.run_module("dmp.experiment.aspect_test")

    #sys.argv[1] = data["config"]
    #runpy.run_module("dmp.experiment.aspect_test")
    # wine_quality_white, 529 pollen
    # change aspect test to have a main function
    # once for each architecture. Do it with a depth thats larger than 2
    # compare results, setting a seed 
    #   check validation loss at the end using float comparison
    #   iterations to converge
    #       with the seed I don't know how effective it will be - if you can track em down.
    #       or run a few times and check to see if the average or SD is similar
    pass

if __name__=="__main__":
    test_aspect_test()