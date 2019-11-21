# Code for Azure Machine Learning Compute - Run-based creation

# Check core SDK version number
import azureml.core

print("SDK version:", azureml.core.VERSION)


from azureml.core import Workspace
ws = Workspace.from_config()


# Set up an experiment 
from azureml.core import Experiment
experiment_name = 'my-experiment'
script_folder= "./"

exp = Experiment(workspace=ws, name=experiment_name)


#<run_temp_compute>
from azureml.core.compute import ComputeTarget, AmlCompute

# First, list the supported VM families for Azure Machine Learning Compute
print(AmlCompute.supported_vmsizes(workspace=ws))

from azureml.core.runconfig import RunConfiguration
# Create a new runconfig object
run_temp_compute = RunConfiguration()

# Signal that you want to use AmlCompute to execute the script
run_temp_compute.target = "amlcompute"

# AmlCompute is created in the same region as your workspace
# Set the VM size for AmlCompute from the list of supported_vmsizes
run_temp_compute.amlcompute.vm_size = 'STANDARD_D2_V2'
#</run_temp_compute>


# Submit the experiment using the run configuration
from azureml.core import ScriptRunConfig

src = ScriptRunConfig(source_directory = script_folder, script = 'train.py', run_config = run_temp_compute)
run = exp.submit(src)
run.wait_for_completion(show_output = True)



