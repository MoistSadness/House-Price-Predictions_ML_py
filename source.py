####################################################################################################################
#
# My first project using machine learning for anything
# Suggested by https://towardsdatascience.com/15-awesome-python-and-data-science-projects-for-2021-and-beyond-64acf7930c20
#
####################################################################################################################

import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd





boston = load_boston()              # Load Dataset

x=boston.data                       # independent variables
y=boston.target                     # target variable

boston_df = pd.DataFrame(x)         # Creating a dataframe of the data set
print()
print(boston_df)                    # Printing Dataframe