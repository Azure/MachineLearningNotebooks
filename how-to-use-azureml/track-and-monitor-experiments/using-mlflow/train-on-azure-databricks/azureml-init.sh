#!/bin/bash
# This script configures the environment to
# 1. Use the given AzureML Workspace with Workspace.from_config()
# 2. Set the default MLflow Tracking Server to be the AzureML managed one

############## START CONFIGURATION #################
# Provide the required AzureML workspace information
region="" # example: westus2
subscriptionId="" # example: bcb65f42-f234-4bff-91cf-9ef816cd9936
resourceGroupName="" # example: dev-rg
workspaceName="" # example: myazuremlws

# Optional config directory
configLocation="/databricks/config.json"
############### END CONFIGURATION #################
# Drop the workspace configuration on the cluster

sudo touch $configLocation
sudo echo {\\"subscription_id\\": \\"${subscriptionId}\\", \\"resource_group\\": \\"${resourceGroupName}\\", \\"workspace_name\\": \\"${workspaceName}\\"} > $configLocation

# Set the MLflow Tracking URI
trackingUri="adbazureml://${region}.experiments.azureml.net/history/v1.0/subscriptions/${subscriptionId}/resourceGroups/${resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}"
sudo echo export MLFLOW_TRACKING_URI=${trackingUri} >> /databricks/spark/conf/spark-env.sh