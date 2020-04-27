import json
import pandas as pd
from sklearn.externals import joblib
from azureml.core.model import Model
import tensorflow as tf


def init():
    global preprocess
    global network
    global scoring_explainer

    # Retrieve the path to the model file using the model name
    # Assume original model is named original_prediction_model
    featurize_path = Model.get_model_path('featurize')
    keras_model_path = Model.get_model_path('keras_model')
    scoring_explainer_path = Model.get_model_path('IBM_attrition_explainer')

    preprocess = joblib.load(featurize_path)
    network = tf.keras.models.load_model(keras_model_path)
    scoring_explainer = joblib.load(scoring_explainer_path)


def run(raw_data):
    # Get predictions and explanations for each data point
    data = pd.read_json(raw_data)
    preprocessed_data = preprocess.transform(data)
    # Make prediction
    predictions = network.predict(preprocessed_data)
    # Retrieve model explanations
    local_importance_values = scoring_explainer.explain(data)
    # You can return any data type as long as it is JSON-serializable
    return {'predictions': predictions.tolist(), 'local_importance_values': local_importance_values}
