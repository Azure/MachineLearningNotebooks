## Azure Machine Learning vignettes

These vignettes are end-to-end tutorials for using Azure Machine Learning SDK for R.

Before running a vignette in RStudio, set the working directory to the folder that contains the vignette file (.Rmd file) in RStudio using `setwd(dirname)` or Session -> Set Working Directory -> To Source File Location. Each vignette assumes that the data and scripts are in the current working directory.

The following vignettes are included:
1. [installation](installation.Rmd): Install the Azure ML SDK for R.
2. [configuration](configuration.Rmd): Set up an Azure ML workspace.
3. [train-and-deploy-to-aci](train-and-deploy-to-aci): Train a caret model and deploy as a web service to Azure Container Instances (ACI).
4. [train-with-tensorflow](train-with-tensorflow/): Train a deep learning TensorFlow model with Azure ML.
5. [hyperparameter-tune-with-keras](hyperparameter-tune-with-keras/): Hyperparameter tune a Keras model using HyperDrive, Azure ML's hyperparameter tuning functionality.
6. [deploy-to-aks](deploy-to-aks/): Production deploy a model as a web service to Azure Kubernetes Service (AKS).

> Before you run these samples, make sure you have an Azure Machine Learning workspace. You can follow the [configuration vignette](../vignettes/configuration.Rmd) to set up a workspace. (You do not need to do this if you are running these examples on an Azure Machine Learning compute instance).

For additional examples on using the R SDK, see the [samples](../samples) folder.