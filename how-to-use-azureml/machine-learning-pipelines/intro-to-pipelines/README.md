# Introduction to Azure Machine Learning Pipelines

The following notebooks provide an introduction to a concept in Azure Machine Learning Pipelines. They will introduce you to core Azure Machine Learning Pipelines features. 
These notebooks below are designed to go in sequence.

1. [aml-pipelines-getting-started.ipynb](https://aka.ms/pl-get-started): Start with this notebook to understand the concepts of using Azure Machine Learning Pipelines. This notebook will show you how to runs steps in parallel and in sequence.
2. [aml-pipelines-with-data-dependency-steps.ipynb](https://aka.ms/pl-data-dep): This notebooks shows how to connect steps in your pipeline using data. Data produced by one step is used by subsequent steps to force an explicit dependency between steps. 
3. [aml-pipelines-publish-and-run-using-rest-endpoint.ipynb](https://aka.ms/pl-pub-rep): Once you are satisfied with your iterative runs in, you could publish your pipeline to get a REST endpoint which could be invoked from non-Pythons clients as well. 
4. [aml-pipelines-data-transfer.ipynb](https://aka.ms/pl-data-trans): This notebook shows how you transfer data between supported datastores.
5. [aml-pipelines-use-databricks-as-compute-target.ipynb](https://aka.ms/pl-databricks): This notebooks shows how you can use Pipelines to send your compute payload to Azure Databricks.
6. [aml-pipelines-use-adla-as-compute-target.ipynb](https://aka.ms/pl-adla): This notebook shows how you can use Azure Data Lake Analytics (ADLA) as a compute target.
7. [aml-pipelines-with-commandstep.ipynb](aml-pipelines-with-commandstep.ipynb): This notebook shows how to use the CommandStep.
8. [aml-pipelines-parameter-tuning-with-hyperdrive.ipynb](https://aka.ms/pl-hyperdrive): HyperDriveStep in Pipelines shows how you can do hyper parameter tuning using Pipelines.
9. [aml-pipelines-how-to-use-azurebatch-to-run-a-windows-executable.ipynb](https://aka.ms/pl-azbatch): AzureBatchStep can be used to run your custom code in AzureBatch cluster.
10. [aml-pipelines-setup-schedule-for-a-published-pipeline.ipynb](https://aka.ms/pl-schedule): Once you publish a Pipeline, you can schedule it to trigger based on an interval or on data change in a defined datastore.
11. [aml-pipelines-with-automated-machine-learning-step.ipynb](https://aka.ms/pl-automl): AutoMLStep in Pipelines shows how you can do automated machine learning using Pipelines.
12. [aml-pipelines-setup-versioned-pipeline-endpoints.ipynb](https://aka.ms/pl-ver-endpoint): This notebook shows how you can setup PipelineEndpoint and submit a Pipeline using the PipelineEndpoint.
13. [aml-pipelines-showcasing-datapath-and-pipelineparameter.ipynb](https://aka.ms/pl-datapath): This notebook showcases how to use DataPath and PipelineParameter in AML Pipeline.
14. [aml-pipelines-how-to-use-pipeline-drafts.ipynb](http://aka.ms/pl-pl-draft): This notebook shows how to use Pipeline Drafts. Pipeline Drafts are mutable pipelines which can be used to submit runs and create Published Pipelines.
15. [aml-pipelines-hot-to-use-modulestep.ipynb](https://aka.ms/pl-modulestep): This notebook shows how to define Module, ModuleVersion and how to use them in an AML Pipeline using ModuleStep.
16. [aml-pipelines-with-notebook-runner-step.ipynb](https://aka.ms/pl-nbrstep): This notebook shows how you can run another notebook as a step in Azure Machine Learning Pipeline.
17. [aml-pipelines-with-commandstep-r.ipynb](aml-pipelines-with-commandstep-r.ipynb): This notebook shows how to use CommandStep to run R scripts.

 ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/README.png)
