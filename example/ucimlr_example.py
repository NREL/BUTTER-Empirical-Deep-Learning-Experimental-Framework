"""
UCIMLR Homepage: https://archive.ics.uci.edu/ml/index.php
Python API: https://pypi.org/project/ucimlr/
Alternate API: https://github.com/tirthajyoti/UCI-ML-API

Name, Data Type, Task, Feature Types, # Observations, # Features
"YearPredictionMSD" Multivariate Regression Real 515345 90
"Beijing Multi-Site Air-Quality Data Data Set" Multivariate, Time-Series Regression Integer, Real 420768 18 2019
"Online Video Characteristics and Transcoding Time Dataset" Multivariate Regression Integer, Real 168286 11 2015
"Metro Interstate Traffic Volume Data Set" Multivariate, Sequential, Time-Series  Regression Integer, Real 48204 9 2019
"Superconductivty Data" Multivariate Regression Real 21263 81 2018
"Appliances energy prediction" Multivariate, Time-Series Regression Real 19735 29 2017
"Pen-Based Recognition of Handwritten Digits" Multivariate Classification Integer 10992 16 1998
"Seoul Bike Sharing Demand" Multivariate Regression Integer, Real 8760 14 2020
"Bias correction of numerical prediction model temperature forecast" Multivariate Regression Real 7750 25 2020


"""

from ucimlr import regression_datasets

abalone = regression_datasets.Abalone('dataset_folder')

print(abalone.type_)
print(abalone.x.shape)
print(abalone.y.shape)