# Deploy R model using AzureML and AzureML Python SDK on Databricks

## 1. Solution Overview

Sample notebook in this repo `notebook.ipynb` provide a simple way to deploy R model using AzureML and AzureML Python SDK on Databrick or other Azure ML compute.

## 1.1 Scope

The scope of this notebook is to deploy R model using AzureML and AzureML Python SDK on Databricks.This notebook assume you have R model is already trained and ready to deploy. This notebook assumes you have pre-processing R script and R model file ( rds ). `rpy2` python package is leveraged to read incoming http requests and respond to the request. Autenthication to AzureML is done using interactive login, but Service priciple can also be used [reference](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.authentication.serviceprincipalauthentication?view=azure-ml-py)

### 1.2 Tools/Libraries used

1. AzureML Create & use software environments in Azure Machine Learning [Link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments)

2. Deploy a model to an Azure Kubernetes Service cluster [Link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-azure-kubernetes-service?tabs=python)

3. R and pandas data frames [Link](https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html)

## 2. How to use this notebook

Import this `notebook.ipynb` file to Databricks and update relevant portions highlighted in the notebook to match your environment.

### 2.1 Prerequisites

This notebook assumes the environment is setup as follows:

1. Databricks workspace is deployed and a cluster to run the notebook is created.

2. R model is trained either on Datbricks cluster or on Azure ML compute.

3. AzureML workspace is created

4. AKS cluster is created and linked to AzureML workspace

5. Service principal details with permission to access AzureML workspace and any other data sources.

### 3. Next Steps

Provide steps to deploy using AzureML CLI
