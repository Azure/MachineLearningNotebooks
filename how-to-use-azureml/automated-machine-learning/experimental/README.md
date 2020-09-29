# Experimental Notebooks for Automated ML
Notebooks listed in this folder are leveraging experimental features. Namespaces or function signitures may change in future SDK releases. The notebooks published here will reflect the latest supported APIs. All of these notebooks can run on a client-only installation of the Automated ML SDK.
The client only installation doesn't contain any of the machine learning libraries, such as scikit-learn, xgboost, or tensorflow, making it much faster to install and is less likely to conflict with any packages in an existing environment. However, since the ML libraries are not available locally, models cannot be downloaded and loaded directly in the client. To replace the functionality of having models locally, these notebooks also demonstrate the ModelProxy feature which will allow you to submit a predict/forecast to the training environment. 

<a name="localconda"></a>
## Setup using a Local Conda environment

To run these notebook on your own notebook server, use these installation instructions.
The instructions below will install everything you need and then start a Jupyter notebook.
If you would like to use a lighter-weight version of the client that does not install all of the machine learning libraries locally, you can leverage the [experimental notebooks.](experimental/README.md)

### 1. Install mini-conda from [here](https://conda.io/miniconda.html), choose 64-bit Python 3.7 or higher.
- **Note**: if you already have conda installed, you can keep using it but it should be version 4.4.10 or later (as shown by: conda -V).  If you have a previous version installed, you can update it using the command: conda update conda.
There's no need to install mini-conda specifically.

### 2. Downloading the sample notebooks
- Download the sample notebooks from [GitHub](https://github.com/Azure/MachineLearningNotebooks) as zip and extract the contents to a local directory.  The automated ML sample notebooks are in the "automated-machine-learning" folder.

### 3. Setup a new conda environment
The **automl_setup_thin_client** script creates a new conda environment, installs the necessary packages, configures the widget and starts a jupyter notebook. It takes the conda environment name as an optional parameter.  The default conda environment name is azure_automl_experimental.  The exact command depends on the operating system.  See the specific sections below for Windows, Mac and Linux.  It can take about 10 minutes to execute.

Packages installed by the **automl_setup** script:
    <ul><li>python</li><li>nb_conda</li><li>matplotlib</li><li>numpy</li><li>cython</li><li>urllib3</li><li>pandas</li><li>azureml-sdk</li><li>azureml-widgets</li><li>pandas-ml</li></ul>

For more details refer to the [automl_env_thin_client.yml](./automl_env_thin_client.yml)
## Windows
Start an **Anaconda Prompt** window, cd to the **how-to-use-azureml/automated-machine-learning/experimental** folder where the sample notebooks were extracted and then run:
```
automl_setup_thin_client
```
## Mac
Install "Command line developer tools" if it is not already installed (you can use the command: `xcode-select --install`).

Start a Terminal windows, cd to the **how-to-use-azureml/automated-machine-learning/experimental** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_thin_client_mac.sh
```

## Linux
cd to the **how-to-use-azureml/automated-machine-learning/experimental** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_thin_client_linux.sh
```

### 4. Running configuration.ipynb
- Before running any samples you next need to run the configuration notebook. Click on [configuration](../../configuration.ipynb) notebook
- Execute the cells in the notebook to Register Machine Learning Services Resource Provider and create a workspace. (*instructions in notebook*)

### 5. Running Samples
- Please make sure you use the Python [conda env:azure_automl_experimental] kernel when trying the sample Notebooks.
- Follow the instructions in the individual notebooks to explore various features in automated ML.

### 6. Starting jupyter notebook manually
To start your Jupyter notebook manually, use:

```
conda activate azure_automl
jupyter notebook
```

or on Mac or Linux:

```
source activate azure_automl
jupyter notebook
```


<a name="samples"></a>
# Automated ML SDK Sample Notebooks

- [auto-ml-regression-model-proxy.ipynb](regression-model-proxy/auto-ml-regression-model-proxy.ipynb)
    - Dataset: Hardware Performance Dataset
    - Simple example of using automated ML for regression
    - Uses azure compute for training
    - Uses ModelProxy for submitting prediction to training environment on azure compute

<a name="documentation"></a>
See [Configure automated machine learning experiments](https://docs.microsoft.com/azure/machine-learning/service/how-to-configure-auto-train) to learn how more about the the settings and features available for automated machine learning experiments.

<a name="pythoncommand"></a>
# Running using python command
Jupyter notebook provides a File / Download as / Python (.py) option for saving the notebook as a Python file.
You can then run this file using the python command.
However, on Windows the file needs to be modified before it can be run.
The following condition must be added to the main code in the file:

    if __name__ == "__main__":

The main code of the file must be indented so that it is under this condition.
