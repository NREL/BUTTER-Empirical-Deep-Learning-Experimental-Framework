from pprint import pprint

import pmlb

'''
PMLB Homepage: https://github.com/EpistasisLab/pmlb
Python Reference: https://epistasislab.github.io/pmlb/python-ref.html



"537_houses" regression, 20640 observations, 8 features, 1 output
"529_pollen" regression, 3848 observations, 4 features, 1 output
"560_bodyfat" regression, 252 observations, 14 variables, 1 output


"adult" classification, 48842 observations, 14 features, 2 classes
"nursery" classification, 12958 observations, 8 features, 7 classes
"ring" classification, 7400 observations, 19 features, 2 classes
"satimage" classification, 6435 observations, 19 features, 6 classes
"cars"  classification, 392 observations, 8 features, 3 classes
"wine_recognition" classification, 178 observations, 13 features, 3 classes
"titanic" classification, 2201 observations, 3 features, 2 classes

'''





X, y = pmlb.fetch_data('mushroom', return_X_y=True)

pprint(X)
pprint(y)
