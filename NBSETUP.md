# Setting up environment

---

To run the notebooks in this repository use one of the two options.

## Option 1: Use Azure Notebooks
Azure Notebooks is a hosted Jupyter-based notebook service in the Azure cloud. Azure Machine Learning Python SDK is already pre-installed in the Azure Notebooks `Python 3.6` kernel.

1. [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://aka.ms/aml-clone-azure-notebooks)
[Import sample notebooks ](https://aka.ms/aml-clone-azure-notebooks) into Azure Notebooks
1. Follow the instructions in the [Configuration](configuration.ipynb) notebook to create and connect to a workspace
1. Open one of the sample notebooks

    **Make sure the Azure Notebook kernel is set to `Python 3.6`** when you open a notebook

    ![set kernel to Python 3.6](images/python36.png)

## **Option 2: Use your own notebook server**

### Quick installation
We recommend you create a Python virtual environment ([Miniconda](https://conda.io/miniconda.html) preferred but [virtualenv](https://virtualenv.pypa.io/en/latest/) works too) and install the SDK in it.
```sh
# install just the base SDK
pip install azureml-sdk

# clone the sample repoistory
git clone https://github.com/Azure/MachineLearningNotebooks.git

# below steps are optional
# install the base SDK and a Jupyter notebook server
pip install azureml-sdk[notebooks]

# install the data prep component
pip install azureml-dataprep

# install model explainability component
pip install azureml-sdk[explain]

# install automated ml components
pip install azureml-sdk[automl]

# install experimental features (not ready for production use)
pip install azureml-sdk[contrib]
```

### Full instructions
[Install the Azure Machine Learning SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python)

Please make sure you start with the [Configuration](configuration.ipynb) notebook to create and connect to a workspace.


### Video walkthrough:

[![Get Started video](images/yt_cover.png)](https://youtu.be/VIsXeTuW3FU)
