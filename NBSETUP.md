# Set up your notebook environment for Azure Machine Learning

To run the notebooks in this repository use one of following options.

## **Option 1: Use Azure Notebooks**
Azure Notebooks is a hosted Jupyter-based notebook service in the Azure cloud. Azure Machine Learning Python SDK is already pre-installed in the Azure Notebooks `Python 3.6` kernel.

1. [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://aka.ms/aml-clone-azure-notebooks)
[Import sample notebooks ](https://aka.ms/aml-clone-azure-notebooks) into Azure Notebooks
1. Follow the instructions in the [Configuration](configuration.ipynb) notebook to create and connect to a workspace
1. Open one of the sample notebooks

    **Make sure the Azure Notebook kernel is set to `Python 3.6`** when you open a notebook by choosing Kernel > Change Kernel > Python 3.6 from the menus.

## **Option 2: Use your own notebook server**

### Quick installation
We recommend you create a Python virtual environment ([Miniconda](https://conda.io/miniconda.html) preferred but [virtualenv](https://virtualenv.pypa.io/en/latest/) works too) and install the SDK in it.
```sh
# install just the base SDK
pip install azureml-sdk

# clone the sample repoistory
git clone https://github.com/Azure/MachineLearningNotebooks.git

# below steps are optional
# install the base SDK, Jupyter notebook server and tensorboard
pip install azureml-sdk[notebooks,tensorboard]

# install model explainability component
pip install azureml-sdk[explain]

# install automated ml components
pip install azureml-sdk[automl]

# install experimental features (not ready for production use)
pip install azureml-sdk[contrib]
```

Note the _extras_ (the keywords inside the square brackets) can be combined. For example:
```sh
# install base SDK, Jupyter notebook and automated ml components
pip install azureml-sdk[notebooks,automl]
```

### Full instructions
[Install the Azure Machine Learning SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python)

Please make sure you start with the [Configuration](configuration.ipynb) notebook to create and connect to a workspace.


### Video walkthrough:

[!VIDEO https://youtu.be/VIsXeTuW3FU]

## **Option 3: Use Docker**

You need to have Docker engine installed locally and running. Open a command line window and type the following command. 

__Note:__ We use version `1.0.10` below as an exmaple, but you can replace that with any available version number you like.

```sh
# clone the sample repoistory
git clone https://github.com/Azure/MachineLearningNotebooks.git

# change current directory to the folder 
# where Dockerfile of the specific SDK version is located.
cd MachineLearningNotebooks/Dockerfiles/1.0.10

# build a Docker image with the a name (azuremlsdk for example) 
# and a version number tag (1.0.10 for example).
# this can take several minutes depending on your computer speed and network bandwidth.
docker build . -t azuremlsdk:1.0.10

# launch the built Docker container which also automatically starts
# a Jupyter server instance listening on port 8887 of the host machine
docker run -it -p 8887:8887 azuremlsdk:1.0.10
```

Now you can point your browser to http://localhost:8887. We recommend that you start from the `configuration.ipynb` notebook at the root directory.

If you need additional Azure ML SDK components, you can either modify the Docker files before you build the Docker images to add additional steps, or install them through command line in the live container after you build the Docker image. For example:

```sh
# install the core SDK and automated ml components
pip install azureml-sdk[automl]

# install the core SDK and model explainability component
pip install azureml-sdk[explain]

# install the core SDK and experimental components
pip install azureml-sdk[contrib]
```
Drag and Drop
The image will be downloaded by Fatkun