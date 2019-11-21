
from azureml.core import Workspace
ws = Workspace.from_config()

from azureml.core.compute import ComputeTarget

# refers to an existing compute resource attached to the workspace!
hdi_compute = ComputeTarget(workspace=ws, name='sherihdi')
    
        
#<run_hdi>
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies


# use pyspark framework
run_hdi = RunConfiguration(framework="pyspark")

# Set compute target to the HDI cluster
run_hdi.target = hdi_compute.name

# specify CondaDependencies object to ask system installing numpy
cd = CondaDependencies()
cd.add_conda_package('numpy')
run_hdi.environment.python.conda_dependencies = cd
#</run_hdi>
print(run_hdi)