# Initial Params
file_name = 'bovespa_indexes.csv'
showData = False # Show the Plotted Data (Matplotlib)
logs = False # Show Processing Step Logs
univariant = True # Only plots at the end if Univariant
trainingFeature = 'DateDist' # (univariant) DateDist, High, Low, Open, Volume
# High, Low and Open are deceivingly accurate, because of high semantic relation to the target class
# *DateDist is the distance (days) from the first registered date in the dataset
testFraction = 0.2 # Recommended: 0.2 | Innacuracy value demonstration: 0.999

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
X = bi.drop(columns=['Adj Close', 'Close', 'Date', 'High', 'Low', 'Open']) if not univariant else bi[[trainingFeature]] # Features: Rest 
# (Date overwritten by DateDist, Close removed because of intrinsic relation with Adj Close)

## Data Visualization
if(showData):
    bi.hist(bins=20, figsize=(10, 8))
    plt.tight_layout()
    plt.show()

## Model Training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testFraction)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

## Model Evaluation

y_pred = pipeline.predict(X_test[:evaluationSampleSize])
modelAccuracy = pipeline.score(X_test, y_test)

print(f"Model Evaluation ({'Univariant' if univariant else 'Multivariant'}{' ['+trainingFeature+']' if univariant else ''})")
print(f"Accuracy: {modelAccuracy}")

## Plot 

Xt_sorted = X_test.sort_values(by=trainingFeature)
yt_pred = pipeline.predict(Xt_sorted)

X_ScatterHandler = X_test if univariant else X_test.iloc[:,0:1]

# Handle the case when univariant is False and you want to plot the selected feature
if univariant:
    X_ScatterHandler = X_test[[trainingFeature]]  # Keep the single feature for plotting
else:
    X_ScatterHandler = X_test[[trainingFeature]]  # Use the specified feature for the x-axis

# Plot
plt.scatter(X_ScatterHandler, y_test, color='black', alpha=0.5, label='Actual')
plt.plot(Xt_sorted[[trainingFeature]], yt_pred, color='blue', linewidth=2, label='Regression Line')
plt.xlabel(f"Training Feature ({trainingFeature})")
plt.ylabel('Adjusted Close Price (Adj Close)')
plt.legend()
plt.title("Linear Regression: Single Feature")
univariant and plt.show()