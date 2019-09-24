# Adding an init script to an Azure Databricks cluster

The [azureml-cluster-init.sh](./azureml-cluster-init.sh) script configures the environment to
1. Use the configured AzureML Workspace with Workspace.from_config()
2. Set the default MLflow Tracking Server to be the AzureML managed one

Modify azureml-cluster-init.sh by providing the values for region, subscriptionId, resourceGroupName, and workspaceName of your target Azure ML workspace in the highlighted section at the top of the script.

To create the Azure Databricks cluster-scoped init script

1. Create the base directory you want to store the init script in if it does not exist.
    ```
    dbutils.fs.mkdirs("dbfs:/databricks/<directory>/")
    ```

2. Create the script by copying the contents of azureml-cluster-init.sh
    ```
    dbutils.fs.put("/databricks/<directory>/azureml-cluster-init.sh","""
    <configured_contents_of_azureml-cluster-init.sh>
    """, True)

3. Check that the script exists.
    ```
    display(dbutils.fs.ls("dbfs:/databricks/<directory>/azureml-cluster-init.sh"))
    ```

1. Configure the cluster to run the script.
    * Using the cluster configuration page
        1. On the cluster configuration page, click the Advanced Options toggle.
        1. At the bottom of the page, click the Init Scripts tab.
        1. In the Destination drop-down, select a destination type. Example: 'DBFS'
        1. Specify a path to the init script.
            ```
            dbfs:/databricks/<directory>/azureml-cluster-init.sh
            ```
        1. Click Add

    * Using the API.
        ```
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
            "destination": "dbfs:/databricks/<directory>/azureml-cluster-init.sh"
            }
        } ]
        }' https://<databricks-instance>/api/2.0/clusters/edit
        ```
