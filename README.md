# Azure Machine Learning service example notebooks

This repository contains example notebooks demonstrating the [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning-service/) Python SDK which allows you to build, train, deploy and manage machine learning solutions using Azure.  The AML SDK allows you the choice of using local or cloud compute resources, while managing and maintaining the complete data science workflow from the cloud.

![Azure ML workflow](https://raw.githubusercontent.com/MicrosoftDocs/azure-docs/master/articles/machine-learning/service/media/overview-what-is-azure-ml/aml.png)

## Quick installation
```sh
pip install azureml-sdk
```
Read more detailed instructions on [how to set up your environment](./NBSETUP.md) using Azure Notebook service, your own Jupyter notebook server, or Docker.

## How to navigate and use the example notebooks?
If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, you should always run the [Configuration](./configuration.ipynb) notebook first when setting up a notebook library on a new machine or in a new environment. It configures your notebook library to connect to an Azure Machine Learning workspace, and sets up your workspace and compute to be used by many of the other examples. 

If you want to...

 * ...try out and explore Azure ML, start with image classification tutorials: [Part 1 (Training)](./tutorials/img-classification-part1-training.ipynb) and [Part 2 (Deployment)](./tutorials/img-classification-part2-deploy.ipynb).
 * ...prepare your data and do automated machine learning, start with regression tutorials: [Part 1 (Data Prep)](./tutorials/regression-part1-data-prep.ipynb) and [Part 2 (Automated ML)](./tutorials/regression-part2-automated-ml.ipynb).
 * ...learn about experimentation and tracking run history, first [train within Notebook](./how-to-use-azureml/training/train-within-notebook/train-within-notebook.ipynb), then try [training on remote VM](./how-to-use-azureml/training/train-on-remote-vm/train-on-remote-vm.ipynb) and [using logging APIs](./how-to-use-azureml/training/logging-api/logging-api.ipynb).
 * ...train deep learning models at scale, first learn about [Machine Learning Compute](./how-to-use-azureml/training/train-on-amlcompute/train-on-amlcompute.ipynb), and then try [distributed hyperparameter tuning](./how-to-use-azureml/training-with-deep-learning/train-hyperparameter-tune-deploy-with-pytorch/train-hyperparameter-tune-deploy-with-pytorch.ipynb) and [distributed training](./how-to-use-azureml/training-with-deep-learning/distributed-pytorch-with-horovod/distributed-pytorch-with-horovod.ipynb).
 * ...deploy models as a realtime scoring service, first learn the basics by [training within Notebook and deploying to Azure Container Instance](./how-to-use-azureml/training/train-within-notebook/train-within-notebook.ipynb), then learn how to [register and manage models, and create Docker images](./how-to-use-azureml/deployment/register-model-create-image-deploy-service/register-model-create-image-deploy-service.ipynb), and [production deploy models on Azure Kubernetes Cluster](./how-to-use-azureml/deployment/production-deploy-to-aks/production-deploy-to-aks.ipynb).
 * ...deploy models as a batch scoring service, first [train a model within Notebook](./how-to-use-azureml/training/train-within-notebook/train-within-notebook.ipynb), learn how to [register and manage models](./how-to-use-azureml/deployment/register-model-create-image-deploy-service/register-model-create-image-deploy-service.ipynb), then [create Machine Learning Compute for scoring compute](./how-to-use-azureml/training/train-on-amlcompute/train-on-amlcompute.ipynb), and [use Machine Learning Pipelines to deploy your model](https://aka.ms/pl-batch-scoring).
 * ...monitor your deployed models, learn about using [App Insights](./how-to-use-azureml/deployment/enable-app-insights-in-production-service/enable-app-insights-in-production-service.ipynb) and [model data collection](./how-to-use-azureml/deployment/enable-data-collection-for-models-in-aks/enable-data-collection-for-models-in-aks.ipynb).

## Tutorials

The [Tutorials](./tutorials) folder contains notebooks for the tutorials described in the [Azure Machine Learning documentation](https://aka.ms/aml-docs).
  
## How to use Azure ML

The [How to use Azure ML](./how-to-use-azureml) folder contains specific examples demonstrating the features of the Azure Machine Learning SDK

- [Training](./how-to-use-azureml/training) - Examples of how to build models using Azure ML's logging and execution capabilities on local and remote compute targets
- [Training with Deep Learning](./how-to-use-azureml/training-with-deep-learning) - Examples demonstrating how to build deep learning models using estimators and parameter sweeps
- [Manage Azure ML Service](./how-to-use-azureml/manage-azureml-service) - Examples how to perform tasks, such as authenticate against Azure ML service in different ways.
- [Automated Machine Learning](./how-to-use-azureml/automated-machine-learning) - Examples using Automated Machine Learning to automatically generate optimal machine learning pipelines and models
- [Machine Learning Pipelines](./how-to-use-azureml/machine-learning-pipelines) - Examples showing how to create and use reusable pipelines for training and batch scoring
- [Deployment](./how-to-use-azureml/deployment) - Examples showing how to deploy and manage machine learning models and solutions
- [Azure Databricks](./how-to-use-azureml/azure-databricks) - Examples showing how to use Azure ML with Azure Databricks
- [Monitor Models](./how-to-use-azureml/monitor-models) - Examples showing how to enable model monitoring services such as DataDrift

---
## Documentation

 * Quickstarts, end-to-end tutorials, and how-tos on the [official documentation site for Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/).
 * [Python SDK reference](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py)
 * Azure ML Data Prep SDK [overview](https://aka.ms/data-prep-sdk), [Python SDK reference](https://aka.ms/aml-data-prep-apiref), and [tutorials and how-tos](https://aka.ms/aml-data-prep-notebooks).

---

## Projects using Azure Machine Learning

Visit following repos to see projects contributed by Azure ML users:

 - [AMLSamples](https://github.com/Azure/AMLSamples) Number of end-to-end examples, including face recognition, predictive maintenance, customer churn and sentiment analysis.
 - [Fine tune natural language processing models using Azure Machine Learning service](https://github.com/Microsoft/AzureML-BERT)
 - [Fashion MNIST with Azure ML SDK](https://github.com/amynic/azureml-sdk-fashion)
 
## Data/Telemetry 
This repository collects usage data and sends it to Mircosoft to help improve our products and services. Read Microsoft's [privacy statement to learn more](https://privacy.microsoft.com/en-US/privacystatement)

To opt out of tracking, please go to the raw markdown or .ipynb files and remove the following line of code:

```sh
    "![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/README.png)"
```
This URL will be slightly different depending on the file. 

 ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/README.png)
