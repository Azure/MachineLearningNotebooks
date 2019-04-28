# Model Deployment with Azure ML service
You can use Azure Machine Learning to package, debug, validate and deploy inference containers to a variety of compute targets. This process is known as "MLOps" (ML operationalization).
For more information please check out this article: https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where

## Get Started
To begin, you will need an ML workspace.
For more information please check out this article: https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace

## Deploy to the cloud
You can deploy to the cloud using the Azure ML CLI or the Azure ML SDK.

### Deploy with the CLI
```
az extension add -n azure-cli-ml
az ml folder attach -w myworkspace -g myresourcegroup
az ml model register -n sklearn_regression_model.pkl -p sklearn_regression_model.pkl -t model.json
az ml model deploy -n acicicd -f model.json --ic inferenceConfig.yml --dc deploymentConfig.yml
```

Here is an [Azure DevOps Pipelines model deployment example](./azure-pipelines-model-deploy.yml)

### Deploy from a notebook
- Notebook example: [model-register-and-deploy](./model-register-and-deploy.ipynb).
