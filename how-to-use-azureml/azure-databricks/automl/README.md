# Adding an init script to an Azure Databricks cluster

The [azureml-cluster-init.sh](./azureml-cluster-init.sh) script configures the environment to
1. Install the latest AutoML library

To create the Azure Databricks cluster-scoped init script

1. Create the base directory you want to store the init script in if it does not exist.
    ```
    dbutils.fs.mkdirs("dbfs:/databricks/init/")
    ```

2. Create the script azureml-cluster-init.sh
    ```
    dbutils.fs.put("/databricks/init/azureml-cluster-init.sh","""
    #!/bin/bash
	set -ex
	/databricks/python/bin/pip install -r https://aka.ms/automl_linux_requirements.txt
    """, True)
    ```

3. Check that the script exists.
    ```
    display(dbutils.fs.ls("dbfs:/databricks/init/azureml-cluster-init.sh"))
    ```

1. Configure the cluster to run the script.
    * Using the cluster configuration page
        1. On the cluster configuration page, click the Advanced Options toggle.
        1. At the bottom of the page, click the Init Scripts tab.
        1. In the Destination drop-down, select a destination type. Example: 'DBFS'
        1. Specify a path to the init script.
            ```
            dbfs:/databricks/init/azureml-cluster-init.sh
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
            "destination": "dbfs:/databricks/init/azureml-cluster-init.sh"
            }
        } ]
        }' https://<databricks-instance>/api/2.0/clusters/edit
        ```
