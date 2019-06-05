**Azure HDInsight**

Azure HDInsight is a fully managed cloud Hadoop & Spark offering the gives
optimized open-source analytic clusters for Spark, Hive, MapReduce, HBase,
Storm, and Kafka. HDInsight Spark clusters provide kernels that you can use with
the Jupyter notebook on [Apache Spark](https://spark.apache.org/) for testing
your applications. 

How Azure HDInsight works with Azure Machine Learning service

-   You can train a model using Spark clusters and deploy the model to ACI/AKS
    from within Azure HDInsight.

-   You can also use [automated machine
    learning](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml) capabilities
    integrated within Azure HDInsight.

You can use Azure HDInsight as a compute target from an [Azure Machine Learning
pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-ml-pipelines).

**Set up your HDInsight cluster**

Create [HDInsight
cluster](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-provision-linux-clusters)

**Quick create: Basic cluster setup**

This article walks you through setup in the [Azure
portal](https://portal.azure.com/), where you can create an HDInsight cluster
using *Quick create* or *Custom*.

![hdinsight create options custom quick create](media/0a235b34c0b881117e51dc31a232dbe1.png)

Follow instructions on the screen to do a basic cluster setup. Details are
provided below for:

-   [Resource group
    name](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-provision-linux-clusters#resource-group-name)

-   [Cluster types and
    configuration](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-provision-linux-clusters#cluster-types)
    (Cluster must be Spark 2.3 (HDI 3.6) or greater)

-   Cluster login and SSH username

-   [Location](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-provision-linux-clusters#location)

**Import the sample HDI notebook in Jupyter**

**Important links:**

Create HDI cluster:
<https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-provision-linux-clusters>

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/azure-hdi/README.png)
