# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.
import os

from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.core.dataset import Dataset
from azureml.train.automl.runtime.automl_explain_utilities import AutoMLExplainerSetupClass, \
    automl_setup_model_explanations, automl_check_model_if_explainable
from azureml.explain.model.mimic.models.lightgbm_model import LGBMExplainableModel
from azureml.explain.model.mimic_wrapper import MimicWrapper
from azureml.automl.core.shared.constants import MODEL_PATH
from azureml.explain.model.scoring.scoring_explainer import TreeScoringExplainer
import joblib

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get workspace from the run context
run = Run.get_context()
ws = run.experiment.workspace

# Get the AutoML run object from the experiment name and the workspace
experiment = Experiment(ws, '<<experiment_name>>')
automl_run = Run(experiment=experiment, run_id='<<run_id>>')

# Check if this AutoML model is explainable
if not automl_check_model_if_explainable(automl_run):
    raise Exception("Model explanations is currently not supported for " + automl_run.get_properties().get(
        'run_algorithm'))

# Download the best model from the artifact store
automl_run.download_file(name=MODEL_PATH, output_file_path='model.pkl')

# Load the AutoML model into memory
fitted_model = joblib.load('model.pkl')

# Get the train dataset from the workspace
train_dataset = Dataset.get_by_name(workspace=ws, name='<<train_dataset_name>>')
# Drop the lablled column to get the training set.
X_train = train_dataset.drop_columns(columns=['<<target_column_name>>'])
y_train = train_dataset.keep_columns(columns=['<<target_column_name>>'], validate=True)

# Get the train dataset from the workspace
test_dataset = Dataset.get_by_name(workspace=ws, name='<<test_dataset_name>>')
# Drop the lablled column to get the testing set.
X_test = test_dataset.drop_columns(columns=['<<target_column_name>>'])

# Setup the class for explaining the AtuoML models
automl_explainer_setup_obj = automl_setup_model_explanations(fitted_model, '<<task>>',
                                                             X=X_train, X_test=X_test,
                                                             y=y_train)

# Initialize the Mimic Explainer
explainer = MimicWrapper(ws, automl_explainer_setup_obj.automl_estimator, LGBMExplainableModel,
                         init_dataset=automl_explainer_setup_obj.X_transform, run=automl_run,
                         features=automl_explainer_setup_obj.engineered_feature_names,
                         feature_maps=[automl_explainer_setup_obj.feature_map],
                         classes=automl_explainer_setup_obj.classes)

# Compute the engineered explanations
engineered_explanations = explainer.explain(['local', 'global'], tag='engineered explanations',
                                            eval_dataset=automl_explainer_setup_obj.X_test_transform)

# Compute the raw explanations
raw_explanations = explainer.explain(['local', 'global'], get_raw=True, tag='raw explanations',
                                     raw_feature_names=automl_explainer_setup_obj.raw_feature_names,
                                     eval_dataset=automl_explainer_setup_obj.X_test_transform)

print("Engineered and raw explanations computed successfully")

# Initialize the ScoringExplainer
scoring_explainer = TreeScoringExplainer(explainer.explainer, feature_maps=[automl_explainer_setup_obj.feature_map])

# Pickle scoring explainer locally
with open('scoring_explainer.pkl', 'wb') as stream:
    joblib.dump(scoring_explainer, stream)

# Upload the scoring explainer to the automl run
automl_run.upload_file('outputs/scoring_explainer.pkl', 'scoring_explainer.pkl')
