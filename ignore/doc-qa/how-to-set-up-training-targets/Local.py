# Code for Local computer and Submit training run sections

# Check core SDK version number
import azureml.core

print("SDK version:", azureml.core.VERSION)

#<run_local>
from azureml.core.runconfig import RunConfiguration

# Edit a run configuration property on the fly.
run_local = RunConfiguration()

run_local.environment.python.user_managed_dependencies = True
#</run_local>

from azureml.core import Workspace
ws = Workspace.from_config()


# Set up an experiment 
# <experiment>
from azureml.core import Experiment
experiment_name = 'my_experiment'

exp = Experiment(workspace=ws, name=experiment_name)
# </experiment>

# Submit the experiment using the run configuration
#<local_submit>
from azureml.core import ScriptRunConfig
import os 

script_folder = os.getcwd()
src = ScriptRunConfig(source_directory = script_folder, script = 'train.py', run_config = run_local)
run = exp.submit(src)
run.wait_for_completion(show_output = True)
#</local_submit>

