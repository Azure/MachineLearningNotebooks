#Adding init script to an Azure Databricks cluster

The azureml-init.sh script configures the environment to
1. Use the given AzureML Workspace with Workspace.from_config()
2. Set the default MLflow Tracking Server to be the AzureML managed one

Modify azureml-init.sh by adding the values for region, subscriptionId, resourceGroupName, and workspaceName.

To create the Azure Databricks cluster-scoped init script

1- Create the base directory you want to store the init script in if it does not exist.
Example: dbutils.fs.mkdirs("dbfs:/databricks/<directory>/")

2- Create the script by copying the contents of azureml-init.sh
dbutils.fs.put("/databricks/<directory>/azureml-init.sh","""
<contents_of_azureml-init.sh>
""", True)

3- Check that the script exists.
Example: display(dbutils.fs.ls("dbfs:/databricks/<directory>/azureml-init.sh"))

4- Configure the cluster to run the script. 
- Using the cluster configuration page 
a) On the cluster configuration page, click the Advanced Options toggle.
b) At the bottom of the page, click the Init Scripts tab.
c) In the Destination drop-down, select a destination type. Example: 'DBFS'
d) Specify a path to the init script. Example: dbfs:/databricks/<directory>/azureml-init.sh
e) Click Add

- Using the API. Example:
curl -n -X POST -H 'Content-Type: application/json' -d '{
  "cluster_id": "<cluster_id>",
  "num_workers": <num_workers>,
  "spark_version": "<spark_version>",
  "node_type_id": "<node_type_id>",
  "cluster_log_conf": {
    "dbfs" : {
      "destination": "dbfs:/cluster-logs"
    }
  },
  "init_scripts": [ {
    "dbfs": {
      "destination": "dbfs:/databricks/<directory>/azureml-init.sh"
    }
  } ]
}' https://<databricks-instance>/api/2.0/clusters/edit