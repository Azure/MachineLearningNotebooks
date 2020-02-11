## Examples to get started with Azure Machine Learning SDK for R

Learn how to use Azure Machine Learning SDK for R for experimentation and model management.

As a pre-requisite, go through the [Installation](vignettes/installation.Rmd) and [Configuration](vignettes/configuration.Rmd) vignettes to first install the package and set up your Azure Machine Learning Workspace unless you are running these examples on an Azure Machine Learning compute instance. Azure Machine Learning compute instances have the Azure Machine Learning SDK pre-installed and your workspace details pre-configured.


Samples
* Deployment
	* [deploy-to-aci](./samples/deployment/deploy-to-aci): Deploy a model as a web service to Azure Container Instances (ACI).
	* [deploy-to-local](./samples/deployment/deploy-to-local): Deploy a model as a web service locally.
* Training
	* [train-on-amlcompute](./samples/training/train-on-amlcompute): Train a model on a remote AmlCompute cluster.
	* [train-on-local](./samples/training/train-on-local): Train a model locally with Docker.

Vignettes
* [deploy-to-aks](./vignettes/deploy-to-aks): Production deploy a model as a web service to Azure Kubernetes Service (AKS).
* [hyperparameter-tune-with-keras](./vignettes/hyperparameter-tune-with-keras): Hyperparameter tune a Keras model using HyperDrive, Azure ML's hyperparameter tuning functionality.
* [train-and-deploy-to-aci](./vignettes/train-and-deploy-to-aci): Train a caret model and deploy as a web service to Azure Container Instances (ACI).
* [train-with-tensorflow](./vignettes/train-with-tensorflow): Train a deep learning TensorFlow model with Azure ML.

Find more information on the [official documentation site for Azure Machine Learning SDK for R](https://azure.github.io/azureml-sdk-for-r/).


### Troubleshooting

- If the following error occurs when submitting an experiment using RStudio:
   ```R
    Error in py_call_impl(callable, dots$args, dots$keywords) : 
     PermissionError: [Errno 13] Permission denied
   ```
  Move the files for your project into a subdirectory and reset the working directory to that directory before re-submitting.
  
  In order to submit an experiment, the Azure ML SDK must create a .zip file of the project directory to send to the service. However,
  the SDK does not have permission to write into the .Rproj.user subdirectory that is automatically created during an RStudio
  session. For this reason, the recommended best practice is to isolate project files into their own directory.
