**Create Azure Databricks Cluster:**

Select New Cluster and fill in following detail:
 - Cluster name: clustername
 - Cluster Mode: **High Concurrency** preferred
 - Databricks Runtime: Any 4.* runtime (NO GPU) or Recommended: 4.3(includes Apache spark 2.3.1, Scala 2.11))
 - Python version: **3**
 - Driver type – you may select a small driver node size (eg. Standard_DS3_v2 0.75 DBU)
 - Worker node VM types: Memory optimized preferred.
	 - Number of concurrent runs should be less than or equal to the number
   of cores in your Databricks cluster.
   -   For a 1 GB numeric only dataset, to do 10 cross validations with run 16 concurrent runs, the minimum usable cluster memory should be 1
   GB X 16 concurrent runs X 3 = 48 GB. This is in addition to what
   Spark itself will use on your cluster.
   -   For text dataset, with featurization (eg. one hot encoding) & cross validation this requirement is much higher. For a 500 MB
   string+numeric dataset, to do 5 cross validation with 4 concurrent
   runs, the minimum usable cluster memory should be 0.5 GB X 4
   concurrent runs X 3 X 5 cross validations X 3 = 90 GB
  - Uncheck _Enable Autoscaling_
- Workers: 2 or higher
- It will take few minutes to create the cluster.

Ensure that the cluster state is running before proceeding further.

**Install Azure ML with Automated ML SDK**

- Select Import library

- Source: Upload Python Egg or PyPI

- PyPi Name: **azureml-sdk[automl_databricks]**

- Click Install Library

- Do not select _Attach automatically to all clusters_. In case you have selected earlier then you can go to your Home folder and deselect it.

- Select the check box _Attach_ next to your cluster name

(More details on attach and detach are here - [https://docs.databricks.com/user-guide/libraries.html#attach-a-library-to-a-cluster](https://docs.databricks.com/user-guide/libraries.html#attach-a-library-to-a-cluster) )

- Ensure that there are no errors until Status changes to _Attached_. It may take a couple of minutes.

**Note** - If you have the old build the please deselect it from cluster’s installed libs > move to trash. Install the new build and restart the cluster. And if still there is an issue then detach and reattach your cluster.

**Now you are run the Automated ML sample notebook on your Azure Databricks cluster.**
