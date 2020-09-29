# 02-create-compute.py
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

ws = Workspace.from_config()

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    cfg = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_D2_V2',
        max_nodes=4,
        idle_seconds_before_scaledown=2400
    )
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, cfg)

cpu_cluster.wait_for_completion(show_output=True)
