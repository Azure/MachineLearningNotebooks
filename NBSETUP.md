# Notebook setup

---

To run the notebooks in this repository use one of these methods:

## Use Azure Notebooks - Jupyter based notebooks in the Azure cloud

1. [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://aka.ms/aml-clone-azure-notebooks)
[Import sample notebooks ](https://aka.ms/aml-clone-azure-notebooks) into Azure Notebooks
1. Follow the instructions in the [Configuration](configuration.ipynb) notebook to create and connect to a workspace
1. Open one of the sample notebooks

    **Make sure the Azure Notebook kernel is set to `Python 3.6`** when you open a notebook

    ![set kernel to Python 3.6](images/python36.png)

## **Use your own notebook server**

Video walkthrough:

[![Get Started video](images/yt_cover.png)](https://youtu.be/VIsXeTuW3FU)

1. Setup a Jupyter Notebook server and [install the Azure Machine Learning SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python)
1. Clone [this repository](https://aka.ms/aml-notebooks)
1. You may need to install other packages for specific notebook
    - For example, to run the Azure Machine Learning Data Prep notebooks, install the extra dataprep SDK:
    ```bash
     pip install azureml-dataprep
    ```

1. Start your notebook server
1. Follow the instructions in the [Configuration](configuration.ipynb) notebook to create and connect to a workspace
1. Open one of the sample notebooks
