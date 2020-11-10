# Azure Machine Learning service example notebooks

> a community-driven repository of examples using mlflow for tracking can be found at https://github.com/Azure/azureml-examples

This repository contains example notebooks demonstrating the [Azure Machine Learning](https://azure.microsoft.com/services/machine-learning-service/) Python SDK which allows you to build, train, deploy and manage machine learning solutions using Azure.  The AML SDK allows you the choice of using local or cloud compute resources, while managing and maintaining the complete data science workflow from the cloud.

![Azure ML Workflow](https://raw.githubusercontent.com/MicrosoftDocs/azure-docs/master/articles/machine-learning/media/concept-azure-machine-learning-architecture/workflow.png)


## Quick installation
```sh
pip install azureml-sdk
```
Read more detailed instructions on [how to set up your environment](./NBSETUP.md) using Azure Notebook service, your own Jupyter notebook server, or Docker.

## How to navigate and use the example notebooks?
If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, you should always run the [Configuration](./configuration.ipynb) notebook first when setting up a notebook library on a new machine or in a new environment. It configures your notebook library to connect to an Azure Machine Learning workspace, and sets up your workspace and compute to be used by many of the other examples. 
This [index](./index.md) should assist in navigating the Azure Machine Learning notebook samples and encourage efficient retrieval of topics and content. 

If you want to...

 * ...try out and explore Azure ML, start with image classification tutorials: [Part 1 (Training)](./tutorials/image-classification-mnist-data/img-classification-part1-training.ipynb) and [Part 2 (Deployment)](./tutorials/image-classification-mnist-data/img-classification-part2-deploy.ipynb).
 * ...learn about experimentation and tracking run history: [track and monitor experiments](./how-to-use-azureml/track-and-monitor-experiments).
 * ...train deep learning models at scale, first learn about [Machine Learning Compute](./how-to-use-azureml/training/train-on-amlcompute/train-on-amlcompute.ipynb), and then try [distributed hyperparameter tuning](./how-to-use-azureml/ml-frameworks/pytorch/train-hyperparameter-tune-deploy-with-pytorch/train-hyperparameter-tune-deploy-with-pytorch.ipynb) and [distributed training](./how-to-use-azureml/ml-frameworks/pytorch/distributed-pytorch-with-horovod/distributed-pytorch-with-horovod.ipynb).
 * ...deploy models as a realtime scoring service, first learn the basics by [deploying to Azure Container Instance](./how-to-use-azureml/deployment/deploy-to-cloud/model-register-and-deploy.ipynb), then learn how to [production deploy models on Azure Kubernetes Cluster](./how-to-use-azureml/deployment/production-deploy-to-aks/production-deploy-to-aks.ipynb).
 * ...deploy models as a batch scoring service: [create Machine Learning Compute for scoring compute](./how-to-use-azureml/training/train-on-amlcompute/train-on-amlcompute.ipynb) and [use Machine Learning Pipelines to deploy your model](https://aka.ms/pl-batch-scoring).
 * ...monitor your deployed models, learn about using [App Insights](./how-to-use-azureml/deployment/enable-app-insights-in-production-service/enable-app-insights-in-production-service.ipynb).

## Tutorials

The [Tutorials](./tutorials) folder contains notebooks for the tutorials described in the [Azure Machine Learning documentation](https://aka.ms/aml-docs).
  
## How to use Azure ML

The [How to use Azure ML](./how-to-use-azureml) folder contains specific examples demonstrating the features of the Azure Machine Learning SDK

- [Training](./how-to-use-azureml/training) - Examples of how to build models using Azure ML's logging and execution capabilities on local and remote compute targets
- [Training with ML and DL frameworks](./how-to-use-azureml/ml-frameworks) - Examples demonstrating how to build and train machine learning models at scale on Azure ML and perform hyperparameter tuning.
- [Manage Azure ML Service](./how-to-use-azureml/manage-azureml-service) - Examples how to perform tasks, such as authenticate against Azure ML service in different ways.
- [Automated Machine Learning](./how-to-use-azureml/automated-machine-learning) - Examples using Automated Machine Learning to automatically generate optimal machine learning pipelines and models
- [Machine Learning Pipelines](./how-to-use-azureml/machine-learning-pipelines) - Examples showing how to create and use reusable pipelines for training and batch scoring
- [Deployment](./how-to-use-azureml/deployment) - Examples showing how to deploy and manage machine learning models and solutions
- [Azure Databricks](./how-to-use-azureml/azure-databricks) - Examples showing how to use Azure ML with Azure Databricks
- [Reinforcement Learning](./how-to-use-azureml/reinforcement-learning) - Examples showing how to train reinforcement learning agents

---
## Documentation

 * Quickstarts, end-to-end tutorials, and how-tos on the [official documentation site for Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/).
 * [Python SDK reference](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py)
 * Azure ML Data Prep SDK [overview](https://aka.ms/data-prep-sdk), [Python SDK reference](https://aka.ms/aml-data-prep-apiref), and [tutorials and how-tos](https://aka.ms/aml-data-prep-notebooks).

---


## Community Repository 
Visit this [community repository](https://github.com/microsoft/MLOps/tree/master/examples) to find useful end-to-end sample notebooks. Also, please follow these [contribution guidelines](https://github.com/microsoft/MLOps/blob/master/contributing.md) when contributing to this repository.   

## Projects using Azure Machine Learning

Visit following repos to see projects contributed by Azure ML users:
 - [Learn about Natural Language Processing best practices using Azure Machine Learning service](https://github.com/microsoft/nlp)
 - [Pre-Train BERT models using Azure Machine Learning service](https://github.com/Microsoft/AzureML-BERT)
 - [Fashion MNIST with Azure ML SDK](https://github.com/amynic/azureml-sdk-fashion)
 - [UMass Amherst Student Samples](https://github.com/katiehouse3/microsoft-azure-ml-notebooks) - A number of end-to-end machine learning notebooks, including machine translation, image classification, and customer churn, created by students in the 696DS course at UMass Amherst.
 
## Data/Telemetry 
This repository collects usage data and sends it to Microsoft to help improve our products and services. Read Microsoft's [privacy statement to learn more](https://privacy.microsoft.com/en-US/privacystatement)

To opt out of tracking, please go to the raw markdown or .ipynb files and remove the following line of code:

```sh
    "![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/README.png)"
```
This URL will be slightly different depending on the file. 

 ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/README.png)
