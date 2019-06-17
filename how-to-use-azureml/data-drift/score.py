import pickle
import json
import numpy
import azureml.train.automl
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.monitoring import ModelDataCollector
import time
import pandas as pd


def init():
    global model, inputs_dc, prediction_dc, feature_names, categorical_features

    print("Model is initialized" + time.strftime("%H:%M:%S"))
    model_path = Model.get_model_path(model_name="driftmodel")
    model = joblib.load(model_path)

    feature_names = ["usaf", "wban", "latitude", "longitude", "station_name", "p_k",
                     "sine_weekofyear", "cosine_weekofyear", "sine_hourofday", "cosine_hourofday",
                     "temperature-7"]

    categorical_features = ["usaf", "wban", "p_k", "station_name"]

    inputs_dc = ModelDataCollector(model_name="driftmodel",
                                   identifier="inputs",
                                   feature_names=feature_names)

    prediction_dc = ModelDataCollector("driftmodel",
                                       identifier="predictions",
                                       feature_names=["temperature"])


def run(raw_data):
    global inputs_dc, prediction_dc

    try:
        data = json.loads(raw_data)["data"]
        data = pd.DataFrame(data)

        # Remove the categorical features as the model expects OHE values
        input_data = data.drop(categorical_features, axis=1)

        result = model.predict(input_data)

        # Collect the non-OHE dataframe
        collected_df = data[feature_names]

        inputs_dc.collect(collected_df.values)
        prediction_dc.collect(result)
        return result.tolist()
    except Exception as e:
        error = str(e)

        print(error + time.strftime("%H:%M:%S"))
        return error
