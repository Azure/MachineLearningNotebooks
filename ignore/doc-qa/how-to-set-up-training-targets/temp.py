from azureml.core import Workspace
ws = Workspace.from_config()

#<amlcompute_temp>
from azureml.core.compute import ComputeTarget, AmlCompute

# First, list the supported VM families for Azure Machine Learning Compute
print(AmlCompute.supported_vmsizes(workspace=ws))
