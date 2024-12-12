# Initial Params
file_name = 'bovespa_indexes.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
dir_path = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
file_path = dir_path + '/../data/' + file_name

# Load the data
bovespa_indexes = pd.read_csv(file_path)

# bovespa_indexes['Date'] = pd.to_datetime(bovespa_indexes['Date'])

print(bovespa_indexes.head())
