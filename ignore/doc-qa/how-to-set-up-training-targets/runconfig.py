# Code for What's a run configuration 

# <run_system_managed>
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

run_system_managed = RunConfiguration()

# Specify the conda dependencies with scikit-learn
run_system_managed.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])
# </run_system_managed>
print(run_system_managed)


# <run_user_managed>
from azureml.core.runconfig import RunConfiguration

run_user_managed = RunConfiguration()
run_user_managed.environment.python.user_managed_dependencies = True

# Choose a specific Python environment by pointing to a Python path. For example: 
# run_config.environment.python.interpreter_path = '/home/ninghai/miniconda3/envs/sdk2/bin/python'
# </run_user_managed>
print(run_user_managed)

