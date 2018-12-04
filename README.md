# Azure Machine Learning service sample notebooks

---

This repository contains example notebooks demonstrating the [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning-service/) Python SDK
which allows you to build, train, deploy and manage machine learning solutions using Azure.  The AML SDK
allows you the choice of using local or cloud compute resources, while managing
and maintaining the complete data science workflow from the cloud.

You can find instructions on setting up notebooks [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python)

You can find full documentation for Azure Machine Learning [here](https://aka.ms/aml-docs)

## Getting Started

These examples will provide you with an effective way to get started using AML.  Once you're familiar with
some of the capabilities, explore the repository for specific topics.

- [Configuration](./configuration.ipynb) configures your notebook library to easily connect to an
    Azure Machine Learning workspace, and sets up your workspace to be used by many of the other examples.  You should
    always run this first when setting up a notebook library on a new machine or in a new environment
- [Train in notebook](./how-to-use-azureml/training/train-within-notebook) shows how to create a model directly in a notebook while recording
    metrics and deploy that model to a test service
- [Train on remote](./how-to-use-azureml/training/train-on-remote-vm) takes the previous example and shows how to create the model on a cloud compute target
- [Production deploy to AKS](./how-to-use-azureml/deployment/production-deploy-to-aks) shows how to create a production grade inferencing webservice

## Tutorials

The [Tutorials](./tutorials) folder contains notebooks for the tutorials described in the [Azure Machine Learning documentation](https://aka.ms/aml-docs)
  
## How to use AML

The [How to use AML](./how-to-use-azureml) folder contains specific examples demonstrating the features of the Azure Machine Learning SDK

- [Training](./how-to-use-azureml/training) - Examples of how to build models using Azure ML's logging and execution capabilities on local and remote compute targets.
- [Training with Deep Learning](./how-to-use-azureml/training-with-deep-learning) - Examples demonstrating how to build deep learning models using estimators and parameter sweeps
- [Automated Machine Learning](./how-to-use-azureml/automated-machine-learning) - Examples using Automated Machine Learning to automatically generate optimal machine learning pipelines and models
- [Machine Learning Pipelines](./how-to-use-azureml/machine-learning-pipelines) - Examples showing how to create and use reusable pipelines for training and batch scoring
- [Deployment](./how-to-use-azureml/deployment) - Examples showing how to deploy and manage machine learning models and solutions
- [Azure Databricks](./how-to-use-azureml/azure-databricks) - Examples showing how to use Azure ML with Azure Databricks
