# Table of Contents
1. [Automated ML Introduction](#introduction)
1. [Setup using Azure Notebooks](#jupyter)
1. [Setup using Azure Databricks](#databricks)
1. [Setup using a Local Conda environment](#localconda)
1. [Automated ML SDK Sample Notebooks](#samples)
1. [Documentation](#documentation)
1. [Running using python command](#pythoncommand)
1. [Troubleshooting](#troubleshooting)

<a name="introduction"></a>
# Automated ML introduction
Automated machine learning (automated ML) builds high quality machine learning models for you by automating model and hyperparameter selection. Bring a labelled dataset that you want to build a model for, automated ML will give you a high quality machine learning model that you can use for predictions.


If you are new to Data Science, automated ML will help you get jumpstarted by simplifying machine learning model building. It abstracts you from needing to perform model selection, hyperparameter selection and in one step creates a high quality trained model for you to use.

If you are an experienced data scientist, automated ML will help increase your productivity by intelligently performing the model and hyperparameter selection for your training and generates high quality models much quicker than manually specifying several combinations of the parameters and running training jobs. Automated ML provides visibility and access to all the training jobs and the performance characteristics of the models to help you further tune the pipeline if you desire.

Below are the three execution environments supported by automated ML.


 <a name="jupyter"></a>
## Setup using Notebook VMs - Jupyter based notebooks from a Azure VM

1. Open the [ML Azure portal](https://ml.azure.com)
1. Select Compute
1. Select Notebook VMs
1. Click New
1. Type a name for the Vm and select a VM type
1. Click Create

<a name="localconda"></a>
## Setup using a Local Conda environment

To run these notebook on your own notebook server, use these installation instructions.
The instructions below will install everything you need and then start a Jupyter notebook.

### 1. Install mini-conda from [here](https://conda.io/miniconda.html), choose 64-bit Python 3.7 or higher.
- **Note**: if you already have conda installed, you can keep using it but it should be version 4.4.10 or later (as shown by: conda -V).  If you have a previous version installed, you can update it using the command: conda update conda.
There's no need to install mini-conda specifically.

### 2. Downloading the sample notebooks
- Download the sample notebooks from [GitHub](https://github.com/Azure/MachineLearningNotebooks) as zip and extract the contents to a local directory.  The automated ML sample notebooks are in the "automated-machine-learning" folder.

### 3. Setup a new conda environment
The **automl_setup** script creates a new conda environment, installs the necessary packages, configures the widget and starts a jupyter notebook. It takes the conda environment name as an optional parameter.  The default conda environment name is azure_automl.  The exact command depends on the operating system.  See the specific sections below for Windows, Mac and Linux.  It can take about 10 minutes to execute.

Packages installed by the **automl_setup** script:
    <ul><li>python</li><li>nb_conda</li><li>matplotlib</li><li>numpy</li><li>cython</li><li>urllib3</li><li>scipy</li><li>scikit-learn</li><li>pandas</li><li>tensorflow</li><li>py-xgboost</li><li>azureml-sdk</li><li>azureml-widgets</li><li>pandas-ml</li></ul>

For more details refer to the [automl_env.yml](./automl_env.yml)
## Windows
Start an **Anaconda Prompt** window, cd to the **how-to-use-azureml/automated-machine-learning** folder where the sample notebooks were extracted and then run:
```
automl_setup
```
## Mac
Install "Command line developer tools" if it is not already installed (you can use the command: `xcode-select --install`).

Start a Terminal windows, cd to the **how-to-use-azureml/automated-machine-learning** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_mac.sh
```

## Linux
cd to the **how-to-use-azureml/automated-machine-learning** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_linux.sh
```

### 4. Running configuration.ipynb
- Before running any samples you next need to run the configuration notebook. Click on [configuration](../../configuration.ipynb) notebook
- Execute the cells in the notebook to Register Machine Learning Services Resource Provider and create a workspace. (*instructions in notebook*)

### 5. Running Samples
- Please make sure you use the Python [conda env:azure_automl] kernel when trying the sample Notebooks.
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

 <a name="databricks"></a>
## Setup using Azure Databricks

**NOTE**: Please create your Azure Databricks cluster as v6.0 (high concurrency preferred) with **Python 3** (dropdown).
**NOTE**: You should at least have contributor access to your Azure subcription to run the notebook.
- Please remove the previous SDK version if there is any and install the latest SDK by installing **azureml-sdk[automl]** as a PyPi library in Azure Databricks workspace.
- You can find the detail Readme instructions at [GitHub](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/azure-databricks).
- Download the sample notebook automl-databricks-local-01.ipynb from [GitHub](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/azure-databricks) and import into the Azure databricks workspace.
- Attach the notebook to the cluster.

<a name="samples"></a>
# Automated ML SDK Sample Notebooks

- [auto-ml-classification-credit-card-fraud.ipynb](classification-credit-card-fraud/auto-ml-classification-credit-card-fraud.ipynb)
    - Dataset: Kaggle's [credit card fraud detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    - Simple example of using automated ML for classification to fraudulent credit card transactions
    - Uses azure compute for training

- [auto-ml-regression.ipynb](regression/auto-ml-regression.ipynb)
    - Dataset: Hardware Performance Dataset
    - Simple example of using automated ML for regression
    - Uses azure compute for training

- [auto-ml-regression-hardware-performance-explanation-and-featurization.ipynb](regression-hardware-performance-explanation-and-featurization/auto-ml-regression-hardware-performance-explanation-and-featurization.ipynb)
    - Dataset: Hardware Performance Dataset
    - Shows featurization and excplanation
    - Uses azure compute for training

- [auto-ml-forecasting-energy-demand.ipynb](forecasting-energy-demand/auto-ml-forecasting-energy-demand.ipynb)
    - Dataset: [NYC energy demand data](forecasting-a/nyc_energy.csv)
    - Example of using automated ML for training a forecasting model

- [auto-ml-classification-credit-card-fraud-local.ipynb](local-run-classification-credit-card-fraud/auto-ml-classification-credit-card-fraud-local.ipynb)
    - Dataset: Kaggle's [credit card fraud detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    - Simple example of using automated ML for classification to fraudulent credit card transactions
    - Uses local compute for training

- [auto-ml-classification-bank-marketing-all-features.ipynb](classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.ipynb)
    - Dataset: UCI's [bank marketing dataset](https://www.kaggle.com/janiobachmann/bank-marketing-dataset)
    - Simple example of using automated ML for classification to predict term deposit subscriptions for a bank
    - Uses azure compute for training

- [auto-ml-forecasting-orange-juice-sales.ipynb](forecasting-orange-juice-sales/auto-ml-forecasting-orange-juice-sales.ipynb)
    - Dataset: [Dominick's grocery sales of orange juice](forecasting-b/dominicks_OJ.csv)
    - Example of training an automated ML forecasting model on multiple time-series

- [auto-ml-forecasting-bike-share.ipynb](forecasting-bike-share/auto-ml-forecasting-bike-share.ipynb)
    - Dataset: forecasting for a bike-sharing
    - Example of training an automated ML forecasting model on multiple time-series

- [auto-ml-forecasting-function.ipynb](forecasting-high-frequency/auto-ml-forecasting-function.ipynb)
    - Example of training an automated ML forecasting model on multiple time-series

- [auto-ml-forecasting-beer-remote.ipynb](forecasting-beer-remote/auto-ml-forecasting-beer-remote.ipynb)
    - Example of training an automated ML forecasting model on multiple time-series
    - Beer Production Forecasting

- [auto-ml-continuous-retraining.ipynb](continuous-retraining/auto-ml-continuous-retraining.ipynb)
    - Continous retraining using Pipelines and Time-Series TabularDataset

- [auto-ml-classification-text-dnn.ipynb](classification-text-dnn/auto-ml-classification-text-dnn.ipynb)
    - Classification with text data using deep learning in AutoML
    - AutoML highlights here include using deep neural networks (DNNs) to create embedded features from text data.
    - Depending on the compute cluster the user provides, AutoML tried out Bidirectional Encoder Representations from Transformers (BERT) when a GPU compute is used.
    - Bidirectional Long-Short Term neural network (BiLSTM) when a CPU compute is used, thereby optimizing the choice of DNN for the uesr's setup.

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

<a name="troubleshooting"></a>
# Troubleshooting
## automl_setup fails
1. On Windows, make sure that you are running automl_setup from an Anconda Prompt window rather than a regular cmd window.  You can launch the "Anaconda Prompt" window by hitting the Start button and typing "Anaconda Prompt".  If you don't see the application "Anaconda Prompt", you might not have conda or mini conda installed.  In that case, you can install it [here](https://conda.io/miniconda.html)
2. Check that you have conda 64-bit installed rather than 32-bit.  You can check this with the command `conda info`.  The `platform` should be `win-64` for Windows or `osx-64` for Mac.
3. Check that you have conda 4.4.10 or later.  You can check the version with the command `conda -V`.  If you have a previous version installed, you can update it using the command: `conda update conda`.
4. On Linux, if the error is `gcc: error trying to exec 'cc1plus': execvp: No such file or directory`, install build essentials using the command `sudo apt-get install build-essential`.
5. Pass a new name as the first parameter to automl_setup so that it creates a new conda environment. You can view existing conda environments using `conda env list` and remove them with `conda env remove -n <environmentname>`.

## automl_setup_linux.sh fails
If automl_setup_linux.sh fails on Ubuntu Linux with the error: `unable to execute 'gcc': No such file or directory`
1. Make sure that outbound ports 53 and 80 are enabled.  On an Azure VM, you can do this from the Azure Portal by selecting the VM and clicking on Networking.
2. Run the command: `sudo apt-get update`
3. Run the command: `sudo apt-get install build-essential --fix-missing`
4. Run `automl_setup_linux.sh` again.

## configuration.ipynb fails
1) For local conda, make sure that you have susccessfully run automl_setup first.
2) Check that the subscription_id is correct.  You can find the subscription_id in the Azure Portal by selecting All Service and then Subscriptions. The characters "<" and ">" should not be included in the subscription_id value.  For example, `subscription_id = "12345678-90ab-1234-5678-1234567890abcd"` has the valid format.
3) Check that you have Contributor or Owner access to the Subscription.
4) Check that the region is one of the supported regions: `eastus2`, `eastus`, `westcentralus`, `southeastasia`, `westeurope`, `australiaeast`, `westus2`, `southcentralus`
5) Check that you have access to the region using the Azure Portal.

## import AutoMLConfig fails after upgrade from before 1.0.76 to 1.0.76 or later
There were package changes in automated machine learning version 1.0.76, which require the previous version to be uninstalled before upgrading to the new version.
If you have manually upgraded from a version of automated machine learning before 1.0.76 to 1.0.76 or later, you may get the error:
`ImportError: cannot import name 'AutoMLConfig'`

This can be resolved by running:
`pip uninstall azureml-train-automl` and then 
`pip install azureml-train-automl`

The automl_setup.cmd script does this automatically.

## workspace.from_config fails
If the call `ws = Workspace.from_config()` fails:
1) Make sure that you have run the `configuration.ipynb` notebook successfully.
2) If you are running a notebook from a folder that is not under the folder where you ran `configuration.ipynb`, copy the folder aml_config and the file config.json that it contains to the new folder.  Workspace.from_config reads the config.json for the notebook folder or it parent folder.
3) If you are switching to a new subscription, resource group, workspace or region, make sure that you run the `configuration.ipynb` notebook again.  Changing config.json directly will only work if the workspace already exists in the specified resource group under the specified subscription.
4) If you want to change the region, please change the workspace, resource group or subscription.  `Workspace.create` will not create or update a workspace if it already exists, even if the region specified is different.

## Sample notebook fails
If a sample notebook fails with an error that property, method or library does not exist:
1) Check that you have selected correct kernel in jupyter notebook.  The kernel is displayed in the top right of the notebook page.  It can be changed using the `Kernel | Change Kernel` menu option.  For Azure Notebooks, it should be `Python 3.6`.  For local conda environments, it should be the conda envioronment name that you specified in automl_setup.  The default is azure_automl.  Note that the kernel is saved as part of the notebook.  So, if you switch to a new conda environment, you will have to select the new kernel in the notebook.
2) Check that the notebook is for the SDK version that you are using.  You can check the SDK version by executing `azureml.core.VERSION` in a jupyter notebook cell.  You can download previous version of the sample notebooks from GitHub by clicking the `Branch` button, selecting the `Tags` tab and then selecting the version.

## Numpy import fails on Windows
Some Windows environments see an error loading numpy with the latest Python version 3.6.8.  If you see this issue, try with Python version 3.6.7.

## Numpy import fails
Check the tensorflow version in the automated ml conda environment. Supported versions are < 1.13. Uninstall tensorflow from the environment if version is >= 1.13
You may check the version of tensorflow and uninstall as follows
1) start a command shell, activate conda environment where automated ml packages are installed
2) enter `pip freeze` and look for `tensorflow` , if found, the version listed should be < 1.13
3) If the listed version is a not a supported version,  `pip uninstall tensorflow` in the command shell and enter y for confirmation.

## Remote run: DsvmCompute.create fails
There are several reasons why the DsvmCompute.create can fail.  The reason is usually in the error message but you have to look at the end of the error message for the detailed reason.  Some common reasons are:
1) `Compute name is invalid, it should start with a letter, be between 2 and 16 character, and only include letters (a-zA-Z), numbers (0-9) and \'-\'.`  Note that underscore is not allowed in the name.
2) `The requested VM size xxxxx is not available in the current region.`  You can select a different region or vm_size.

## Remote run: Unable to establish SSH connection
Automated ML uses the SSH protocol to communicate with remote DSVMs.  This defaults to port 22.  Possible causes for this error are:
1) The DSVM is not ready for SSH connections.  When DSVM creation completes, the DSVM might still not be ready to acceept SSH connections.  The sample notebooks have a one minute delay to allow for this.
2) Your Azure Subscription may restrict the IP address ranges that can access the DSVM on port 22.  You can check this in the Azure Portal by selecting the Virtual Machine and then clicking Networking.  The Virtual Machine name is the name that you provided in the notebook plus 10 alpha numeric characters to make the name unique.  The Inbound Port Rules define what can access the VM on specific ports.  Note that there is a priority priority order.  So, a Deny entry with a low priority number will override a Allow entry with a higher priority number.

## Remote run: setup iteration fails
This is often an issue with the `get_data` method.
1) Check that the `get_data` method is valid by running it locally.
2) Make sure that `get_data` isn't referring to any local files.  `get_data` is executed on the remote DSVM.  So, it doesn't have direct access to local data files.  Instead you can store the data files with DataStore.  See [auto-ml-remote-execution-with-datastore.ipynb](remote-execution-with-datastore/auto-ml-remote-execution-with-datastore.ipynb)
3) You can get to the error log for the setup iteration by clicking the `Click here to see the run in Azure portal` link, click `Back to Experiment`, click on the highest run number and then click on Logs.

## Remote run: disk full
Automated ML creates files under /tmp/azureml_runs for each iteration that it runs.  It creates a folder with the iteration id.  For example: AutoML_9a038a18-77cc-48f1-80fb-65abdbc33abe_93.  Under this, there is a azureml-logs folder, which contains logs.  If you run too many iterations on the same DSVM, these files can fill the disk.
You can delete the files under /tmp/azureml_runs or just delete the VM and create a new one.
If your get_data downloads files, make sure the delete them or they can use disk space as well.
When using DataStore, it is good to specify an absolute path for the files so that they are downloaded just once.  If you specify a relative path, it will download a file for each iteration.

## Remote run: Iterations fail and the log contains "MemoryError"
This can be caused by insufficient memory on the DSVM.  Automated ML loads all training data into memory.  So, the available memory should be more than the training data size.
If you are using a remote DSVM, memory is needed for each concurrent iteration.  The max_concurrent_iterations setting specifies the maximum concurrent iterations.  For example, if the training data size is 8Gb and max_concurrent_iterations is set to 10, the minimum memory required is at least 80Gb.
To resolve this issue, allocate a DSVM with more memory or reduce the value specified for max_concurrent_iterations.

## Remote run: Iterations show as "Not Responding" in the RunDetails widget.
This can be caused by too many concurrent iterations for a remote DSVM.  Each concurrent iteration usually takes 100% of a core when it is running.  Some iterations can use multiple cores.  So, the max_concurrent_iterations setting should always be less than the number of cores of the DSVM.
To resolve this issue, try reducing the value specified for the max_concurrent_iterations setting.
