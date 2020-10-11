# 01-create-workspace.py
from azureml.core import Workspace

# Example locations: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.
ws = Workspace.create(name='<my_workspace_name>',
                      subscription_id='<azure-subscription-id>',
                      resource_group='<myresourcegroup>',
                      create_resource_group=True,
                      location='<NAME_OF_REGION>')

# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')
