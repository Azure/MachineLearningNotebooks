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

---
## Documentation

 * Quickstarts, end-to-end tutorials, and how-tos on the [official documentation site for Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/).
 * [Python SDK reference](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py)
 * Azure ML Data Prep SDK [overview](https://aka.ms/data-prep-sdk), [Python SDK reference](https://aka.ms/aml-data-prep-apiref), and [tutorials and how-tos](https://aka.ms/aml-data-prep-notebooks).

---

## Projects using Azure Machine Learning

Visit following repos to see projects contributed by Azure ML users:

 - [Fine tune natural language processing models using Azure Machine Learning service](https://github.com/Microsoft/AzureML-BERT)
 - [Fashion MNIST with Azure ML SDK](https://github.com/amynic/azureml-sdk-fashion)
 
 ## Azure Machine Learning Resources & Links
## Product Documentation
- [Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/)
- [Azure Machine Learning Studio](https://docs.microsoft.com/en-us/azure/machine-learning/studio/)

## Product Team Blogs
- [What’s new in Azure Machine Learning service](https://aka.ms/aml-blog-whats-new)
- [Announcing automated ML capability in Azure Machine Learning](https://aka.ms/aml-blog-automl)
- [Experimentation using Azure Machine Learning](https://aka.ms/aml-blog-experimentation)
- [Azure AI – Making AI real for business](https://aka.ms/aml-blog-overview)

## Community Blogs
- [Power Bat – How Spektacom is Powering the Game of Cricket with Microsoft AI](https://blogs.technet.microsoft.com/machinelearning/2018/10/11/power-bat-how-spektacom-is-powering-the-game-of-cricket-with-microsoft-ai/)

## Ignite 2018 Public Preview Launch Sessions
- [AI with Azure Machine Learning services: Simplifying the data science process](https://myignite.techcommunity.microsoft.com/sessions/66248)
- [AI TechTalk: Azure Machine Learning SDK - a walkthrough](https://myignite.techcommunity.microsoft.com/sessions/66265) 
- [AI for an intelligent cloud and intelligent edge: Discover, deploy, and manage with Azure ML services](https://myignite.techcommunity.microsoft.com/sessions/65389)
- [Generating high quality models efficiently using Automated ML and Hyperparameter Tuning](https://myignite.techcommunity.microsoft.com/sessions/66245)
- [AI for pros: Deep learning with PyTorch using the Azure Data Science Virtual Machine and scaling training with Azure ML](https://myignite.techcommunity.microsoft.com/sessions/66244)

## Get-started Videos on YouTube
- [Get started with Python SDK](https://youtu.be/VIsXeTuW3FU)
- [Get started from Azure Portal](https://youtu.be/lCkYUHV86Mk)


## Third Party Articles
- [Azure’s new machine learning features embrace Python](https://www.infoworld.com/article/3306840/azure/azures-new-machine-learning-features-embrace-python.html) (InfoWorld)
- [How to use Azure ML in Windows 10](https://www.infoworld.com/article/3308381/azure/how-to-use-azure-ml-in-windows-10.html) (InfoWorld)
- [How Azure ML Streamlines Cloud-based Machine Learning](https://thenewstack.io/how-the-azure-ml-streamlines-cloud-based-machine-learning/) (The New Stack)
- [Facebook launches PyTorch 1.0 with integrations for Google Cloud, AWS, and Azure Machine Learning](https://venturebeat.com/2018/10/02/facebook-launches-pytorch-1-0-integrations-for-google-cloud-aws-and-azure-machine-learning/) (VentureBeat)
- [How Microsoft Uses Machine Learning to Help You Build Machine Learning Pipelines](https://towardsdatascience.com/how-microsoft-uses-machine-learning-to-help-you-build-machine-learning-pipelines-be75f710613b) (Towards Data Science)
- [Microsoft's Machine Learning Tools for Developers Get Smarter](https://techcrunch.com/2018/09/24/microsofts-machine-learning-tools-for-developers-get-smarter/) (TechCrunch)
- [Microsoft introduces Azure service to automatically build AI models](https://venturebeat.com/2018/09/24/microsoft-introduces-azure-service-to-automatically-build-ai-models/) (VentureBeat)

## Community Projects
- [Use Papermill with Azure ML](https://github.com/jreynolds01/papermill_execution_azureml/)
- [Fashion MNIST](https://github.com/amynic/azureml-sdk-fashion)
- Keras on Databricks
- [Samples from CSS](https://github.com/Azure/AMLSamples)


## Azure Machine Learning Studio Resources
- [A-Z Machine Learning using Azure Machine Learning (AzureML)](https://www.udemy.com/machine-learning-using-azureml/)
- [Machine Learning In The Cloud With Azure Machine Learning](https://www.udemy.com/machine-learning-in-the-cloud-with-azure-machine-learning/)
- [How to Become A Data Scientist Using Azure Machine Learning](https://www.udemy.com/azure-machine-learning-introduction/)
- [Learn Azure Machine Learning from scratch](https://www.udemy.com/learn-azure-machine-learning-from-scratch/)
- [Azure Machine Learning Studio PowerShell Module](https://aka.ms/amlps)

## Forum Help
- [Azure Machine Learning service](https://social.msdn.microsoft.com/Forums/en-US/home?forum=AzureMachineLearningService)
- [Azure Machine Learning Studio](https://social.msdn.microsoft.com/forums/azure/en-US/home?forum=MachineLearning)


## Data/Telemetry 
This repository collects usage data and sends it to Mircosoft to help improve our products and services. Read Microsoft's [privacy statement to learn more](https://privacy.microsoft.com/en-US/privacystatement)

To opt out of tracking, please go to the raw markdown or .ipynb files and remove the following line of code:

```sh
    "![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/README.png)"
```
This URL will be slightly different depending on the file. 


 ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/README.png)
