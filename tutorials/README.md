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
| [Use automated machine learning to predict taxi fares](https://docs.microsoft.com/azure/machine-learning/tutorial-auto-train-models) | Train a regression model to predict taxi fares using Automated Machine Learning. | [regression-part2-automated-ml.ipynb](regression-automl-nyc-taxi-data/regression-automated-ml.ipynb) | Regression | Automated ML 

## Advanced Samples

The following tutorials are intended to provide examples of more advanced feature in Azure Machine Learning.

| Tutorial | Description | Notebook | Task | Framework | 
| --- | --- | --- | --- | --- |
| [Build an Azure Machine Learning pipeline for batch scoring](https://docs.microsoft.com/azure/machine-learning/tutorial-pipeline-batch-scoring-classification) | Create an Azure Machine Learning pipeline to run batch scoring image classification jobs | [tutorial-pipeline-batch-scoring-classification.ipynb](machine-learning-pipelines-advanced/tutorial-pipeline-batch-scoring-classification.ipynb) | Image Classification | TensorFlow
Complete these tutorials to learn how to train and deploy models using Azure Machine Learning services and Python SDK. These Notebooks accompany the tutorial articles for:

For additional documentation and resources, see the [official documentation site for Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/).

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/tutorials/README.png)