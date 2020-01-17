import json
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model


def init():

    global original_model
    global scoring_explainer

    # Retrieve the path to the model file using the model name
    # Assume original model is named original_prediction_model
    original_model_path = Model.get_model_path('amlcompute_deploy_model')
    scoring_explainer_path = Model.get_model_path('IBM_attrition_explainer')

    original_model = joblib.load(original_model_path)
    scoring_explainer = joblib.load(scoring_explainer_path)


def run(raw_data):
    # Get predictions and explanations for each data point
    data = pd.read_json(raw_data)
    # Make prediction
    predictions = original_model.predict(data)
    # Retrieve model explanations
    local_importance_values = scoring_explainer.explain(data)
    # You can return any data type as long as it is JSON-serializable
    return {'predictions': predictions.tolist(), 'local_importance_values': local_importance_values}
