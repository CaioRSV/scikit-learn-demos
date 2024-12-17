# Initial Params
file_name = 'bovespa_indexes.csv'
showData = False # Show the Plotted Data (Matplotlib)
logs = False # Show Processing Step Logs

evaluationSampleSize = 30

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
dir_path = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
file_path = dir_path + '/../data/' + file_name

logs and print("Loading Datasets...")

# Load the data
bi = pd.read_csv(file_path) # Bovespa Indexes
if(logs):
    print("-")
    print("Bovespa Indexes Data")
    print(f"{bi.shape[1]} features, {bi.shape[0]} rows");
    print("-")

## Data Preprocessing
# Date Codification
bi['Date'] = pd.to_datetime(bi['Date']); # Assure DateTime format
bi['DateDist'] = bi['Date'].rank(method='dense').astype(int) # Distance from first date
logs and print(bi.iloc[0:10, 1:].values)

# Symbol Codification
symbolList = np.unique(bi['Symbol'])
symbolMapping = {symbol: idx for idx, symbol in enumerate(symbolList)}
bi['Symbol'] = bi['Symbol'].map(symbolMapping)

## Feature Separation
y = bi['Adj Close'] # Target Value: Adj Close
X = bi.drop(columns=['Adj Close', 'Close', 'Date']) # Features: Rest 
# (Date overwritten by DateDist, Close removed because of intrinsic relation with Adj Close)

## Data Visualization
if(showData):
    bi.hist(bins=20, figsize=(10, 8))
    plt.tight_layout()
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.015)

## Model Training

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

## Model Evaluation

y_pred = pipeline.predict(X_test[:evaluationSampleSize])
modelAccuracy = pipeline.score(X_test, y_test)

print(f"Accuracy: {modelAccuracy}")
print("---")

pred = pipeline.predict(X_test[:evaluationSampleSize])
print(f"Predicted Class:")
print(pred)
print("Real class")
print(y_test[:evaluationSampleSize])


print(len(
    X['DateDist'].tolist()
));
print(len(
    y.tolist()
))

X_single = X[['DateDist']] 

X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2)
pipeline.fit(X_train, y_train)

X_sorted = X_single.sort_values(by='DateDist')
y_pred_line = pipeline.predict(X_sorted)

plt.scatter(X_single, y, color='black', alpha=0.5, label='Actual')
plt.plot(X_sorted, y_pred_line, color='blue', linewidth=2, label='Regression Line')
plt.xlabel('Date Distance (DateDist)')
plt.ylabel('Adjusted Close Price (Adj Close)')
plt.legend()
plt.title("Linear Regression: Single Feature")
plt.show()