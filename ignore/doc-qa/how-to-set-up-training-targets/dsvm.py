# Code for Remote virtual machines

compute_target_name = "attach-dsvm"

#<run_dsvm>  
import azureml.core
from azureml.core.runconfig import RunConfiguration, DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies

run_dsvm = RunConfiguration(framework = "python")

# Set the compute target to the Linux DSVM
run_dsvm.target = compute_target_name 

# Use Docker in the remote VM
run_dsvm.environment.docker.enabled = True

# Use the CPU base image 
# To use GPU in DSVM, you must also use the GPU base Docker image "azureml.core.runconfig.DEFAULT_GPU_IMAGE"
run_dsvm.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE
print('Base Docker image is:', run_dsvm.environment.docker.base_image)

# Prepare the Docker and conda environment automatically when they're used for the first time 
run_dsvm.prepare_environment = True

# Specify the CondaDependencies object
run_dsvm.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])
#</run_dsvm>