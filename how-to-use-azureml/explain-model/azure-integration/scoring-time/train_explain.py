# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper

from azureml.core.run import Run
from interpret.ext.blackbox import TabularExplainer
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient
from azureml.interpret.scoring.scoring_explainer import LinearScoringExplainer, save

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# get the IBM employee attrition dataset
outdirname = 'dataset.6.21.19'
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
zipfilename = outdirname + '.zip'
urlretrieve('https://publictestdatasets.blob.core.windows.net/data/' + zipfilename, zipfilename)
with zipfile.ZipFile(zipfilename, 'r') as unzip:
    unzip.extractall('.')
attritionData = pd.read_csv('./WA_Fn-UseC_-HR-Employee-Attrition.csv')

# dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)
attritionData = attritionData.drop(['Over18'], axis=1)
# since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)

# converting target variables from string to numerical values
target_map = {'Yes': 1, 'No': 0}
attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
target = attritionData["Attrition_numerical"]

attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

# creating dummy columns for each categorical feature
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# store the numerical columns
numerical = attritionXData.columns.difference(categorical)

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]

categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

# append classifier to preprocessing pipeline
clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

# get the run this was submitted from to interact with run history
run = Run.get_context()

# create an explanation client to store the explanation (contrib API)
client = ExplanationClient.from_run(run)

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(attritionXData,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)

# write x_test out as a pickle file for later visualization
x_test_pkl = 'x_test.pkl'
with open(x_test_pkl, 'wb') as file:
    joblib.dump(value=x_test, filename=os.path.join(OUTPUT_DIR, x_test_pkl))
run.upload_file('x_test_ibm.pkl', os.path.join(OUTPUT_DIR, x_test_pkl))

# preprocess the data and fit the classification model
clf.fit(x_train, y_train)
model = clf.steps[-1][1]

# save model for use outside the script
model_file_name = 'log_reg.pkl'
with open(model_file_name, 'wb') as file:
    joblib.dump(value=clf, filename=os.path.join(OUTPUT_DIR, model_file_name))

# register the model with the model management service for later use
run.upload_file('original_model.pkl', os.path.join(OUTPUT_DIR, model_file_name))
original_model = run.register_model(model_name='amlcompute_deploy_model',
                                    model_path='original_model.pkl')

# create an explainer to validate or debug the model
tabular_explainer = TabularExplainer(model,
                                     initialization_examples=x_train,
                                     features=attritionXData.columns,
                                     classes=["Not leaving", "leaving"],
                                     transformations=transformations)

# explain overall model predictions (global explanation)
# passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# more data (e.g. x_train) will likely lead to higher accuracy, but at a time cost
global_explanation = tabular_explainer.explain_global(x_test)

# uploading model explanation data for storage or visualization
comment = 'Global explanation on classification model trained on IBM employee attrition dataset'
client.upload_model_explanation(global_explanation, comment=comment)

# also create a lightweight explainer for scoring time
scoring_explainer = LinearScoringExplainer(tabular_explainer)
# pickle scoring explainer locally
save(scoring_explainer, directory=OUTPUT_DIR, exist_ok=True)

# register scoring explainer
run.upload_file('IBM_attrition_explainer.pkl', os.path.join(OUTPUT_DIR, 'scoring_explainer.pkl'))
scoring_explainer_model = run.register_model(model_name='IBM_attrition_explainer',
                                             model_path='IBM_attrition_explainer.pkl')
