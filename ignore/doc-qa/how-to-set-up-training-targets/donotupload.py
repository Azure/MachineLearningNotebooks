
from azureml.core import Workspace
ws = Workspace.from_config()

from azureml.core.compute import ComputeTarget, HDInsightCompute
from azureml.exceptions import ComputeTargetException

try:
    # if you want to connect using SSH key instead of username/password you can provide parameters private_key_file and private_key_passphrase
    attach_config = HDInsightCompute.attach_configuration(address='sheri2-ssh.azurehdinsight.net', 
                                                          ssh_port=22, 
                                                          username='sshuser', 
                                                          password='ChangePassw)rd12')
    hdi_compute = ComputeTarget.attach(workspace=ws, 
                                       name='sherihdi2', 
                                       attach_configuration=attach_config)

except ComputeTargetException as e:
    print("Caught = {}".format(e.message))
    hdi_compute = ComputeTarget(workspace=ws, name='sherihdi')
    
        
hdi_compute.wait_for_completion(show_output=True)

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