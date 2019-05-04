# Introduction to Azure Machine Learning Pipelines

The following notebooks provide an introduction to a concept in Azure Machine Learning Pipelines. They will introduce you to core Azure Machine Learning Pipelines features. 
These notebooks below are designed to go in sequence.

1. [aml-pipelines-getting-started.ipynb](https://aka.ms/pl-get-started): Start with this notebook to understand the concepts of using Azure Machine Learning Pipelines. This notebook will show you how to runs steps in parallel and in sequence.
2. [aml-pipelines-with-data-dependency-steps.ipynb](https://aka.ms/pl-data-dep): This notebooks shows how to connect steps in your pipeline using data. Data produced by one step is used by subsequent steps to force an explicit dependency between steps. 
3. [aml-pipelines-publish-and-run-using-rest-endpoint.ipynb](https://aka.ms/pl-pub-rep): Once you are satisfied with your iterative runs in, you could publish your pipeline to get a REST endpoint which could be invoked from non-Pythons clients as well. 
4. [aml-pipelines-data-transfer.ipynb](https://aka.ms/pl-data-trans): This notebook shows how you transfer data between supported datastores.
5. [aml-pipelines-use-databricks-as-compute-target.ipynb](https://aka.ms/pl-databricks): This notebooks shows how you can use Pipelines to send your compute payload to Azure Databricks.
6. [aml-pipelines-use-adla-as-compute-target.ipynb](https://aka.ms/pl-adla): This notebook shows how you can use Azure Data Lake Analytics (ADLA) as a compute target.
7. [aml-pipelines-how-to-use-estimatorstep.ipynb](https://aka.ms/pl-estimator): This notebook shows how to use the EstimatorStep.
8. [aml-pipelines-parameter-tuning-with-hyperdrive.ipynb](https://aka.ms/pl-hyperdrive): HyperDriveStep in Pipelines shows how you can do hyper parameter tuning using Pipelines.
9. [aml-pipelines-how-to-use-azurebatch-to-run-a-windows-executable.ipynb](https://aka.ms/pl-azbatch): AzureBatchStep can be used to run your custom code in AzureBatch cluster.
10. [aml-pipelines-setup-schedule-for-a-published-pipeline.ipynb](https://aka.ms/pl-schedule): Once you publish a Pipeline, you can schedule it to trigger based on an interval or on data change in a defined datastore. 


 ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/README.png)
