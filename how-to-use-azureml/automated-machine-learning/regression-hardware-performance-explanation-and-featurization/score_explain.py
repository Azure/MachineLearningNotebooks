import json
import numpy as np
import pandas as pd
import os
import pickle
import azureml.train.automl
import azureml.explain.model
from azureml.train.automl.automl_explain_utilities import AutoMLExplainerSetupClass, automl_setup_model_explanations
from sklearn.externals import joblib
from azureml.core.model import Model


def init():

    global automl_model
    global scoring_explainer

    # Retrieve the path to the model file using the model name
    # Assume original model is named original_prediction_model
    automl_model_path = Model.get_model_path('automl_model')
    scoring_explainer_path = Model.get_model_path('scoring_explainer')

    automl_model = joblib.load(automl_model_path)
    scoring_explainer = joblib.load(scoring_explainer_path)


def run(raw_data):
    # Get predictions and explanations for each data point
    data = pd.read_json(raw_data, orient='records')
    # Make prediction
    predictions = automl_model.predict(data)
    # Setup for inferencing explanations
    automl_explainer_setup_obj = automl_setup_model_explanations(automl_model,
                                                                 X_test=data, task='regression')
    # Retrieve model explanations for engineered explanations
    engineered_local_importance_values = scoring_explainer.explain(automl_explainer_setup_obj.X_test_transform)
    # Retrieve model explanations for raw explanations
    raw_local_importance_values = scoring_explainer.explain(automl_explainer_setup_obj.X_test_transform, get_raw=True)
    # You can return any data type as long as it is JSON-serializable
    return {'predictions': predictions.tolist(),
            'engineered_local_importance_values': engineered_local_importance_values,
            'raw_local_importance_values': raw_local_importance_values}
