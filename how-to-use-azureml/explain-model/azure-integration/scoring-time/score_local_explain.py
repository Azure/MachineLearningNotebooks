import json
import numpy as np
import pandas as pd
import os
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model


def init():

    global original_model
    global scoring_explainer

    # Retrieve the path to the model file using the model name
    # Assume original model is named original_prediction_model
    original_model_path = Model.get_model_path('local_deploy_model')
    scoring_explainer_path = Model.get_model_path('IBM_attrition_explainer')

    # Load the original model into the environment
    original_model = joblib.load(original_model_path)
    # Load the scoring explainer into the environment
    scoring_explainer = joblib.load(scoring_explainer_path)


def run(raw_data):
    # Get predictions and explanations for each data point
    data = pd.read_json(raw_data)
    # Make prediction
    predictions = original_model.predict(data)
    # Retrieve model explanations
    local_importance_values = scoring_explainer.explain(data)
    # Retrieve the feature names, which we may want to return to the user.
    # Note: you can also get the raw_features and engineered_features
    # by calling scoring_explainer.raw_features and
    # scoring_explainer.engineered_features but you may need to pass
    # the raw or engineered feature names in the ScoringExplainer
    # constructor, depending on if you are using feature maps or
    # transformations on the original explainer.
    features = scoring_explainer.features
    # You can return any data type as long as it is JSON-serializable
    return {'predictions': predictions.tolist(),
            'local_importance_values': local_importance_values,
            'features': features}
