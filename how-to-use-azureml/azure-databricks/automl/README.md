# Automated ML introduction
Automated machine learning (automated ML) builds high quality machine learning models for you by automating model and hyperparameter selection. Bring a labelled dataset that you want to build a model for, automated ML will give you a high quality machine learning model that you can use for predictions.


If you are new to Data Science, automated ML will help you get jumpstarted by simplifying machine learning model building. It abstracts you from needing to perform model selection, hyperparameter selection and in one step creates a high quality trained model for you to use.

If you are an experienced data scientist, automated ML will help increase your productivity by intelligently performing the model and hyperparameter selection for your training and generates high quality models much quicker than manually specifying several combinations of the parameters and running training jobs. Automated ML provides visibility and access to all the training jobs and the performance characteristics of the models to help you further tune the pipeline if you desire.

# Install Instructions using Azure Databricks :

#### For Databricks non ML runtime 7.1(scala 2.21, spark 3.0.0) and up, Install Automated Machine Learning sdk by adding and running the following command as the first cell of your notebook. This will install AutoML dependencies specific for your notebook.

%pip install --upgrade --force-reinstall -r https://aka.ms/automl_linux_requirements.txt


#### For Databricks non ML runtime 7.0 and lower, Install Automated Machine Learning sdk using init script as shown below before running the notebook.**

**Create the Azure Databricks cluster-scoped init script 'azureml-cluster-init.sh' as below

1. Create the base directory you want to store the init script in if it does not exist.
    ```
    dbutils.fs.mkdirs("dbfs:/databricks/init/")
    ```

2. Create the script azureml-cluster-init.sh
    ```
    dbutils.fs.put("/databricks/init/azureml-cluster-init.sh","""
    #!/bin/bash
	set -ex
	/databricks/python/bin/pip install --upgrade --force-reinstall -r https://aka.ms/automl_linux_requirements.txt
    """, True)
    ```

3. Check that the script exists.
    ```
    display(dbutils.fs.ls("dbfs:/databricks/init/azureml-cluster-init.sh"))
    ```

**Install libraries to cluster using init script 'azureml-cluster-init.sh' created in previous step

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
