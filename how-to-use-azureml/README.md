## Examples to get started with Azure Machine Learning service

Learn how to use Azure Machine Learning services for experimentation and model management.

As a pre-requisite, run the [configuration Notebook](../configuration.ipynb) notebook first to set up your Azure ML Workspace. Then, run the notebooks in following recommended order.

* [train-within-notebook](./training/train-within-notebook): Train a model while tracking run history, and learn how to deploy the model as web service to Azure Container Instance.
* [train-on-local](./training/train-on-local): Learn how to submit a run to local computer and use Azure ML managed run configuration.
* [train-on-amlcompute](./training/train-on-amlcompute): Use a 1-n node Azure ML managed compute cluster for remote runs on Azure CPU or GPU infrastructure.
* [train-on-remote-vm](./training/train-on-remote-vm): Use Data Science Virtual Machine as a target for remote runs.
* [logging-api](./track-and-monitor-experiments/logging-api): Learn about the details of logging metrics to run history.
* [production-deploy-to-aks](./deployment/production-deploy-to-aks) Deploy a model to production at scale on Azure Kubernetes Service.
* [enable-app-insights-in-production-service](./deployment/enable-app-insights-in-production-service) Learn how to use App Insights with production web service.
 
Find quickstarts, end-to-end tutorials, and how-tos on the [official documentation site for Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/).
