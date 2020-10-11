# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from sklearn import datasets
from sklearn.linear_model import Ridge
from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
import joblib
import os
import numpy as np

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

boston_data = datasets.load_boston()

run = Run.get_context()
client = ExplanationClient.from_run(run)

X_train, X_test, y_train, y_test = train_test_split(boston_data.data,
                                                    boston_data.target,
                                                    test_size=0.2,
                                                    random_state=0)
# write x_test out as a pickle file for later visualization
x_test_pkl = 'x_test.pkl'
with open(x_test_pkl, 'wb') as file:
    joblib.dump(value=X_test, filename=os.path.join(OUTPUT_DIR, x_test_pkl))
run.upload_file('x_test_boston_housing.pkl', os.path.join(OUTPUT_DIR, x_test_pkl))


alpha = 0.5
# Use Ridge algorithm to create a regression model
reg = Ridge(alpha)
model = reg.fit(X_train, y_train)

preds = reg.predict(X_test)
run.log('alpha', alpha)

model_file_name = 'ridge_{0:.2f}.pkl'.format(alpha)
# save model in the outputs folder so it automatically get uploaded
with open(model_file_name, 'wb') as file:
    joblib.dump(value=reg, filename=os.path.join(OUTPUT_DIR,
                                                 model_file_name))

# register the model
run.upload_file('original_model.pkl', os.path.join('./outputs/', model_file_name))
original_model = run.register_model(model_name='model_explain_model_on_amlcomp',
                                    model_path='original_model.pkl')

# Explain predictions on your local machine
tabular_explainer = TabularExplainer(model, X_train, features=boston_data.feature_names)

# Explain overall model predictions (global explanation)
# Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# x_train can be passed as well, but with more examples explanations it will
# take longer although they may be more accurate
global_explanation = tabular_explainer.explain_global(X_test)

# Uploading model explanation data for storage or visualization in webUX
# The explanation can then be downloaded on any compute
comment = 'Global explanation on regression model trained on boston dataset'
client.upload_model_explanation(global_explanation, comment=comment, model_id=original_model.id)
