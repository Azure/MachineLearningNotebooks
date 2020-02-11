## Azure Machine Learning samples
These samples are short code examples for using Azure Machine Learning SDK for R. If you are new to the R SDK, we recommend that you first take a look at the more detailed end-to-end [vignettes](../vignettes).

Before running a sample in RStudio, set the working directory to the folder that contains the sample script in RStudio using `setwd(dirname)` or Session -> Set Working Directory -> To Source File Location. Each vignette assumes that the data and scripts are in the current working directory.

1. [train-on-amlcompute](training/train-on-amlcompute): Train a model on a remote AmlCompute cluster.
2. [train-on-local](training/train-on-local): Train a model locally with Docker.
2. [deploy-to-aci](deployment/deploy-to-aci): Deploy a model as a web service to Azure Container Instances (ACI).
3. [deploy-to-local](deployment/deploy-to-local): Deploy a model as a web service locally.

> Before you run these samples, make sure you have an Azure Machine Learning workspace. You can follow the [configuration vignette](../vignettes/configuration.Rmd) to set up a workspace. (You do not need to do this if you are running these examples on an Azure Machine Learning compute instance).
