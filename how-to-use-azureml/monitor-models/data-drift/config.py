# imports
import pickle
from datetime import datetime
from azureml.opendatasets import NoaaIsdWeather
from sklearn.linear_model import LinearRegression

# get weather dataset
start = datetime(2019, 1, 1)
end = datetime(2019, 1, 14)
isd = NoaaIsdWeather(start, end)

# convert to pandas dataframe and filter down
df = isd.to_pandas_dataframe().fillna(0)
df = df[df['stationName'].str.contains('FLORIDA', regex=True, na=False)]

# features for training
X_features = ['latitude', 'longitude', 'temperature', 'windAngle', 'windSpeed']
y_features = ['elevation']

# write the training dataset to csv
training_dataset = df[X_features + y_features]
training_dataset.to_csv('training.csv', index=False)

# train the model
X = training_dataset[X_features]
y = training_dataset[y_features]
model = LinearRegression().fit(X, y)

# save the model as a .pkl file
with open('elevation-regression-model.pkl', 'wb') as f:
    pickle.dump(model, f)
