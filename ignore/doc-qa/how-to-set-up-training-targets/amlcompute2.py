# Code for Azure Machine Learning Compute - Persistent compute

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

#<cpu_cluster>
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpucluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # To use a different region for the compute, add a location='<region>' parameter
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)
#</cpu_cluster>

#<run_amlcompute>
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

# Create a new runconfig object
run_amlcompute = RunConfiguration()

# Use the cpu_cluster you created above. 
run_amlcompute.target = cpu_cluster

# Enable Docker
run_amlcompute.environment.docker.enabled = True

# Set Docker base image to the default CPU-based image
run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE

# Use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_amlcompute.environment.python.user_managed_dependencies = False

# Specify CondaDependencies obj, add necessary packages
run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])
#</run_amlcompute>

# Submit the experiment using the run configuration
#<amlcompute_submit>
from azureml.core import ScriptRunConfig

src = ScriptRunConfig(source_directory = script_folder, script = 'train.py', run_config = run_amlcompute)
run = exp.submit(src)
run.wait_for_completion(show_output = True)
#</amlcompute_submit>
