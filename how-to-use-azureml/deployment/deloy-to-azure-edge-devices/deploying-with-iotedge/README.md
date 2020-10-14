# Model Deployment with Azure ML service to Azure stack edge using Iot edge
You can use Azure Machine Learning to package, debug, validate and deploy inference containers to a variety of compute targets. This process is known as "MLOps" (ML operationalization). Here we will show you how you can deploy a model from cloud to Azure stack edge device using IoT Edge.
For more information please check out this article: https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-machine-learning-edge-04-train-model

## Get Started
To begin, you will need an ML workspace.
For more information please check out this article: https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace
you also need to get you Azure stacck edge setup and run a sample a gpu sample as : https://docs.microsoft.com/en-us/azure/databox-online/azure-stack-edge-gpu-deploy-sample-module-marketplace


## Deploy to the Azure stack edge
You can deploy to the Azure stack edge as 
- Notebook example: [model-register-and-deploy](./production-deploy-to-ase-gpu.ipynb).