# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from azureml.core.run import Run
import joblib
import os
import shap
import xgboost

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

run = Run.get_context()

# get a dataset on income prediction
X, y = shap.datasets.adult()

# train an XGBoost model (but any other tree model type should work)
model = xgboost.XGBClassifier()
model.fit(X, y)

explainer = shap.explainers.GPUTree(model, X)
X_shap = X[:100]
shap_values = explainer(X_shap)

print("computed shap values:")
print(shap_values)

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
