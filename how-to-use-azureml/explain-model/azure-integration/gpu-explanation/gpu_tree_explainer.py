# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from azureml.core.run import Run
from azureml.interpret import ExplanationClient
from interpret_community.adapter import ExplanationAdapter
import joblib
import os
import shap
import xgboost

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

run = Run.get_context()
client = ExplanationClient.from_run(run)

# get a dataset on income prediction
X, y = shap.datasets.adult()
features = X.columns.values

# train an XGBoost model (but any other tree model type should work)
model = xgboost.XGBClassifier()
model.fit(X, y)

explainer = shap.explainers.GPUTree(model, X)
X_shap = X[:100]
shap_values = explainer(X_shap)

print("computed shap values:")
print(shap_values)

# Use the explanation adapter to convert the importances into an interpret-community
# style explanation which can be uploaded to AzureML or visualized in the
# ExplanationDashboard widget
adapter = ExplanationAdapter(features, classification=True)
global_explanation = adapter.create_global(shap_values.values, X_shap, expected_values=shap_values.base_values)

# write X_shap out as a pickle file for later visualization
x_shap_pkl = 'x_shap.pkl'
with open(x_shap_pkl, 'wb') as file:
    joblib.dump(value=X_shap, filename=os.path.join(OUTPUT_DIR, x_shap_pkl))
run.upload_file('x_shap_adult_census.pkl', os.path.join(OUTPUT_DIR, x_shap_pkl))

model_file_name = 'xgboost_.pkl'
# save model in the outputs folder so it automatically gets uploaded
with open(model_file_name, 'wb') as file:
    joblib.dump(value=model, filename=os.path.join(OUTPUT_DIR,
                                                   model_file_name))

# register the model
run.upload_file('xgboost_model.pkl', os.path.join('./outputs/', model_file_name))
original_model = run.register_model(model_name='xgboost_with_gpu_tree_explainer',
                                    model_path='xgboost_model.pkl')

# Uploading model explanation data for storage or visualization in webUX
# The explanation can then be downloaded on any compute
comment = 'Global explanation on classification model trained on adult census income dataset'
client.upload_model_explanation(global_explanation, comment=comment, model_id=original_model.id)
