# Azure Machine Learning Tutorials

Azure Machine Learning, a cloud-based environment you can use to train, deploy, automate, manage, and track ML models.

Azure Machine Learning can be used for any kind of machine learning, from classical ML to supervised, unsupervised, and deep learning.

This folder contains a collection of Jupyter Notebooks with the code used in accompanying step-by-step tutorials.

## Set up your environment.

If you are using an Azure Machine Learning Notebook VM, everything is already set up for you. Otherwise, see the [get started creating your first ML experiment with the Python SDK tutorial](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup).

## Introductory Samples

The following tutorials are intended to provide an introductory overview of Azure Machine Learning.

| Tutorial | Description | Notebook | Task | Framework | 
| --- | --- | --- | --- | --- |
| [Train your first ML Model](https://docs.microsoft.com/azure/machine-learning/tutorial-1st-experiment-sdk-train) | Learn the foundational design patterns in Azure Machine Learning and train a scikit-learn model based on a diabetes data set. | [tutorial-quickstart-train-model.ipynb](create-first-ml-experiment/tutorial-1st-experiment-sdk-train.ipynb) | Regression | Scikit-Learn
| [Train an image classification model](https://docs.microsoft.com/azure/machine-learning/tutorial-train-models-with-aml) | Train a scikit-learn image classification model. | [img-classification-part1-training.ipynb](image-classification-mnist-data/img-classification-part1-training.ipynb) | Image Classification | Scikit-Learn
| [Deploy an image classification model](https://docs.microsoft.com/azure/machine-learning/tutorial-deploy-models-with-aml) | Deploy a scikit-learn image classification model to Azure Container Instances. | [img-classification-part2-deploy.ipynb](image-classification-mnist-data/img-classification-part2-deploy.ipynb) | Image Classification | Scikit-Learn
| [Deploy an encrypted inferencing service](https://docs.microsoft.com/azure/machine-learning/tutorial-deploy-models-with-aml) |Deploy an image classification model for encrypted inferencing in Azure Container Instances | [img-classification-part3-deploy-encrypted.ipynb](image-classification-mnist-data/img-classification-part3-deploy-encrypted.ipynb) | Image Classification | Scikit-Learn
| [Use automated machine learning to predict taxi fares](https://docs.microsoft.com/azure/machine-learning/tutorial-auto-train-models) | Train a regression model to predict taxi fares using Automated Machine Learning. | [regression-part2-automated-ml.ipynb](regression-automl-nyc-taxi-data/regression-automated-ml.ipynb) | Regression | Automated ML 
| Azure ML in 10 minutes (Compute instance required) |Learn how to run an image classification model, track model metrics, and deploy a model in 10 minutes. | [quickstart-azureml-in-10mins.ipynb](compute-instance-quickstarts/quickstart-azureml-in-10mins/quickstart-azureml-in-10mins.ipynb) | Image Classification | Scikit-Learn |
| Get started with Azure ML Job Submission (Compute instance required) |Learn how to use the Azure Machine Learning Python SDK to submit batch jobs. | [quickstart-azureml-python-sdk.ipynb](compute-instance-quickstarts/quickstart-azureml-python-sdk/quickstart-azureml-python-sdk.ipynb) | Image Classification | Scikit-Learn |
| Get started with Automated ML (Compute instance required) | Learn how to use Automated ML for Fraud classification. | [quickstart-azureml-automl.ipynb](compute-instance-quickstarts/quickstart-azureml-automl/quickstart-azureml-automl.ipynb) | Classification | Automated ML |


## Advanced Samples

The following tutorials are intended to provide examples of more advanced feature in Azure Machine Learning.

| Tutorial | Description | Notebook | Task | Framework | 
| --- | --- | --- | --- | --- |
| [Build an Azure Machine Learning pipeline for batch scoring](https://docs.microsoft.com/azure/machine-learning/tutorial-pipeline-batch-scoring-classification) | Create an Azure Machine Learning pipeline to run batch scoring image classification jobs | [tutorial-pipeline-batch-scoring-classification.ipynb](machine-learning-pipelines-advanced/tutorial-pipeline-batch-scoring-classification.ipynb) | Image Classification | TensorFlow

For additional documentation and resources, see the [official documentation site for Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/).

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/tutorials/README.png)