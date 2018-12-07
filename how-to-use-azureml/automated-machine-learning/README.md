# Table of Contents
1. [Automated ML Introduction](#introduction)
1. [Running samples in Azure Notebooks](#jupyter)
1. [Running samples in Azure Databricks](#databricks)
1. [Running samples in a Local Conda environment](#localconda)
1. [Automated ML SDK Sample Notebooks](#samples)
1. [Documentation](#documentation)
1. [Running using python command](#pythoncommand)
1. [Troubleshooting](#troubleshooting)

<a name="introduction"></a>
# Automated ML introduction
Automated machine learning (automated ML) builds high quality machine learning models for you by automating model and hyperparameter selection. Bring a labelled dataset that you want to build a model for, automated ML will give you a high quality machine learning model that you can use for predictions.


If you are new to Data Science, AutoML will help you get jumpstarted by simplifying machine learning model building. It abstracts you from needing to perform model selection, hyperparameter selection and in one step creates a high quality trained model for you to use.

If you are an experienced data scientist, AutoML will help increase your productivity by intelligently performing the model and hyperparameter selection for your training and generates high quality models much quicker than manually specifying several combinations of the parameters and running training jobs. AutoML provides visibility and access to all the training jobs and the performance characteristics of the models to help you further tune the pipeline if you desire.

Below are the three execution environments supported by AutoML.


 <a name="jupyter"></a>
## Running samples in Azure Notebooks - Jupyter based notebooks in the Azure cloud

1. [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://aka.ms/aml-clone-azure-notebooks)
[Import sample notebooks ](https://aka.ms/aml-clone-azure-notebooks) into Azure Notebooks.
1. Follow the instructions in the [configuration](configuration.ipynb) notebook to create and connect to a workspace.
1. Open one of the sample notebooks.

 <a name="databricks"></a>
## Running samples in Azure Databricks

**NOTE**: Please create your Azure Databricks cluster as v4.x (high concurrency preferred) with **Python 3** (dropdown).
**NOTE**: You should at least have contributor access to your Azure subcription to run the notebook.
- Please remove the previous SDK version if there is any and install the latest SDK by installing **azureml-sdk[automl_databricks]** as a PyPi library in Azure Databricks workspace.
- Download the sample notebook 16a.auto-ml-classification-local-azuredatabricks from [GitHub](https://github.com/Azure/MachineLearningNotebooks) and import into the Azure databricks workspace.
- Attach the notebook to the cluster.

<a name="localconda"></a>
## Running samples in a Local Conda environment

To run these notebook on your own notebook server, use these installation instructions.

The instructions below will install everything you need and then start a Jupyter notebook.  To start your Jupyter notebook manually, use:

```
conda activate azure_automl
jupyter notebook
```

or on Mac:

```
source activate azure_automl
jupyter notebook
```


### 1. Install mini-conda from [here](https://conda.io/miniconda.html), choose Python 3.7 or higher.
- **Note**: if you already have conda installed, you can keep using it but it should be version 4.4.10 or later (as shown by: conda -V).  If you have a previous version installed, you can update it using the command: conda update conda.
There's no need to install mini-conda specifically.

### 2. Downloading the sample notebooks
- Download the sample notebooks from [GitHub](https://github.com/Azure/MachineLearningNotebooks) as zip and extract the contents to a local directory.  The AutoML sample notebooks are in the "automl" folder.

### 3. Setup a new conda environment
The **automl/automl_setup** script creates a new conda environment, installs the necessary packages, configures the widget and starts a jupyter notebook.
It takes the conda environment name as an optional parameter.  The default conda environment name is azure_automl.  The exact command depends on the operating system.  See the specific sections below for Windows, Mac and Linux.  It can take about 10 minutes to execute.
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
- Before running any samples you next need to run the configuration notebook. Click on configuration.ipynb notebook
- Execute the cells in the notebook to Register Machine Learning Services Resource Provider and create a workspace. (*instructions in notebook*)

### 5. Running Samples
- Please make sure you use the Python [conda env:azure_automl] kernel when trying the sample Notebooks.
- Follow the instructions in the individual notebooks to explore various features in AutoML

<a name="samples"></a>
# Automated ML SDK Sample Notebooks
- [configuration.ipynb](configuration.ipynb)
    - Create new Azure ML Workspace
    - Save Workspace configuration file

- [auto-ml-classification.ipynb](classification/auto-ml-classification.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Simple example of using Auto ML for classification
    - Uses local compute for training

- [auto-ml-regression.ipynb](regression/auto-ml-regression.ipynb)
    - Dataset: scikit learn's [diabetes dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
    - Simple example of using Auto ML for regression
    - Uses local compute for training

- [auto-ml-remote-execution.ipynb](remote-execution/auto-ml-remote-execution.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Example of using Auto ML for classification using a remote linux DSVM for training
    - Parallel execution of iterations
    - Async tracking of progress
    - Cancelling individual iterations or entire run
    - Retrieving models for any iteration or logged metric
    - Specify automl settings as kwargs

- [auto-ml-remote-batchai.ipynb](remote-batchai/auto-ml-remote-batchai.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Example of using automated ML for classification using a remote Batch AI compute for training
    - Parallel execution of iterations
    - Async tracking of progress
    - Cancelling individual iterations or entire run
    - Retrieving models for any iteration or logged metric
    - Specify automl settings as kwargs

- [auto-ml-remote-attach.ipynb](remote-attach/auto-ml-remote-attach.ipynb)
    - Dataset: Scikit learn's [20newsgroup](http://scikit-learn.org/stable/datasets/twenty_newsgroups.html)
    - handling text data with preprocess flag
    - Reading data from a blob store for remote executions
    - using pandas dataframes for reading data

- [auto-ml-missing-data-blacklist-early-termination.ipynb](missing-data-blacklist-early-termination/auto-ml-missing-data-blacklist-early-termination.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Blacklist certain pipelines
    - Specify a target metrics to indicate stopping criteria
    - Handling Missing Data in the input

- [auto-ml-sparse-data-train-test-split.ipynb](sparse-data-train-test-split/auto-ml-sparse-data-train-test-split.ipynb)
    - Dataset: Scikit learn's [20newsgroup](http://scikit-learn.org/stable/datasets/twenty_newsgroups.html)
    - Handle sparse datasets
    - Specify custom train and validation set

- [auto-ml-exploring-previous-runs.ipynb](exploring-previous-runs/auto-ml-exploring-previous-runs.ipynb)
    - List all projects for the workspace
    - List all AutoML Runs for a given project
    - Get details for a AutoML Run. (Automl settings, run widget & all metrics)
    - Download fitted pipeline for any iteration

- [auto-ml-remote-execution-with-datastore.ipynb](remote-execution-with-datastore/auto-ml-remote-execution-with-datastore.ipynb)
    - Dataset: Scikit learn's [20newsgroup](http://scikit-learn.org/stable/datasets/twenty_newsgroups.html)
    - Download the data and store it in DataStore.

- [auto-ml-classification-with-deployment.ipynb](classification-with-deployment/auto-ml-classification-with-deployment.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Simple example of using Auto ML for classification
    - Registering the model
    - Creating Image and creating aci service
    - Testing the aci service

- [auto-ml-sample-weight.ipynb](sample-weight/auto-ml-sample-weight.ipynb)
    - How to specifying sample_weight
    - The difference that it makes to test results

- [auto-ml-dataprep.ipynb](dataprep/auto-ml-dataprep.ipynb)
    - Using DataPrep for reading data

- [auto-ml-dataprep-remote-execution.ipynb](dataprep-remote-execution/auto-ml-dataprep-remote-execution.ipynb)
    - Using DataPrep for reading data with remote execution

- [auto-ml-classification-local-azuredatabricks.ipynb](classification-local-azuredatabricks/auto-ml-classification-local-azuredatabricks.ipynb)
    - Dataset: scikit learn's [digit dataset](https://innovate.burningman.org/datasets-page/)
    - Example of using AutoML for classification using Azure Databricks as the platform for training

- [auto-ml-classification_with_tensorflow.ipynb](classification_with_tensorflow/auto-ml-classification_with_tensorflow.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Simple example of using Auto ML for classification with whitelisting tensorflow models.checkout
    - Uses local compute for training

- [auto-ml-forecasting-a.ipynb](forecasting-a/auto-ml-forecasting-a.ipynb)
    - Dataset: [NYC energy demand data](forecasting-a/nyc_energy.csv)
    - Example of using AutoML for training a forecasting model

- [auto-ml-forecasting-b.ipynb](forecasting-b/auto-ml-forecasting-b.ipynb)
    - Dataset: [Dominick's grocery sales of orange juice](forecasting-b/dominicks_OJ.csv)
    - Example of training an AutoML forecasting model on multiple time-series

<a name="documentation"></a>
# Documentation
## Table of Contents
1. [Automated ML Settings ](#automlsettings)
1. [Cross validation split options](#cvsplits)
1. [Get Data Syntax](#getdata)
1. [Data pre-processing and featurization](#preprocessing)

<a name="automlsettings"></a>
## Automated ML Settings

|Property|Description|Default|
|-|-|-|
|**primary_metric**|This is the metric that you want to optimize.<br><br> Classification supports the following primary metrics <br><i>accuracy</i><br><i>AUC_weighted</i><br><i>average_precision_score_weighted</i><br><i>norm_macro_recall</i><br><i>precision_score_weighted</i><br><br> Regression supports the following primary metrics <br><i>spearman_correlation</i><br><i>normalized_root_mean_squared_error</i><br><i>r2_score</i><br><i>normalized_mean_absolute_error</i><br><i>normalized_root_mean_squared_log_error</i>| Classification: accuracy <br><br> Regression: spearman_correlation
|**iteration_timeout_minutes**|Time limit in minutes for each iteration|None|
|**iterations**|Number of iterations. In each iteration trains the data with a specific pipeline.  To get the best result, use at least 100. |100|
|**n_cross_validations**|Number of cross validation splits|None|
|**validation_size**|Size of validation set as percentage of all training samples|None|
|**max_concurrent_iterations**|Max number of iterations that would be executed in parallel|1|
|**preprocess**|*True/False* <br>Setting this to *True* enables preprocessing <br>on the input to handle missing data, and perform some common feature extraction<br>*Note: If input data is Sparse you cannot use preprocess=True*|False|
|**max_cores_per_iteration**| Indicates how many cores on the compute target would be used to train a single pipeline.<br> You can set it to *-1* to use all cores|1|
|**experiment_exit_score**|*double* value indicating the target for *primary_metric*. <br> Once the target is surpassed the run terminates|None|
|**blacklist_models**|*Array* of *strings* indicating models to ignore for Auto ML from the list of models.|None|
|**whitelist_models**|*Array* of *strings* use only models listed for Auto ML from the list of models..|None|
 <a name="cvsplits"></a>
## List of models for white list/blacklist
**Classification**
<br><i>LogisticRegression</i>
<br><i>SGD</i>
<br><i>MultinomialNaiveBayes</i>
<br><i>BernoulliNaiveBayes</i>
<br><i>SVM</i>
<br><i>LinearSVM</i>
<br><i>KNN</i>
<br><i>DecisionTree</i>
<br><i>RandomForest</i>
<br><i>ExtremeRandomTrees</i>
<br><i>LightGBM</i>
<br><i>GradientBoosting</i>
<br><i>TensorFlowDNN</i>
<br><i>TensorFlowLinearClassifier</i>
<br><br>**Regression**
<br><i>ElasticNet</i>
<br><i>GradientBoosting</i>
<br><i>DecisionTree</i>
<br><i>KNN</i>
<br><i>LassoLars</i>
<br><i>SGD</i>
<br><i>RandomForest</i>
<br><i>ExtremeRandomTrees</i>
<br><i>LightGBM</i>
<br><i>TensorFlowLinearRegressor</i>
<br><i>TensorFlowDNN</i>

## Cross validation split options
### K-Folds Cross Validation
Use *n_cross_validations* setting to specify the number of cross validations. The training data set will be randomly split into *n_cross_validations* folds of equal size. During each cross validation round, one of the folds will be used for validation of the model trained on the remaining folds. This process repeats for *n_cross_validations* rounds until each fold is used once as validation set. Finally, the average scores accross all *n_cross_validations* rounds will be reported, and the corresponding model will be retrained on the whole training data set.

### Monte Carlo Cross Validation (a.k.a. Repeated Random Sub-Sampling)
Use *validation_size* to specify the percentage of the training data set that should be used for validation, and use *n_cross_validations* to specify the number of cross validations. During each cross validation round, a subset of size *validation_size* will be randomly selected for validation of the model trained on the remaining data. Finally, the average scores accross all *n_cross_validations* rounds will be reported, and the corresponding model will be retrained on the whole training data set.

### Custom train and validation set
You can specify seperate train and validation set either through the get_data() or directly to the fit method.

<a name="getdata"></a>
## get_data() syntax
The *get_data()* function can be used to return a dictionary with these values:

|Key|Type|Dependency|Mutually Exclusive with|Description|
|:-|:-|:-|:-|:-|
|X|Pandas Dataframe or Numpy Array|y|data_train, label, columns|All features to train with|
|y|Pandas Dataframe or Numpy Array|X|label|Label data to train with.  For classification, this should be an array of integers. |
|X_valid|Pandas Dataframe or Numpy Array|X, y, y_valid|data_train, label|*Optional* All features to validate with.  If this is not specified, X is split between train and validate|
|y_valid|Pandas Dataframe or Numpy Array|X, y, X_valid|data_train, label|*Optional* The label data to validate with.  If this is not specified, y is split between train and validate|
|sample_weight|Pandas Dataframe or Numpy Array|y|data_train, label, columns|*Optional* A weight value for each label. Higher values indicate that the sample is more important.|
|sample_weight_valid|Pandas Dataframe or Numpy Array|y_valid|data_train, label, columns|*Optional* A weight value for each validation label. Higher values indicate that the sample is more important.  If this is not specified, sample_weight is split between train and validate|
|data_train|Pandas Dataframe|label|X, y, X_valid, y_valid|All data (features+label) to train with|
|label|string|data_train|X, y, X_valid, y_valid|Which column in data_train represents the label|
|columns|Array of strings|data_train||*Optional* Whitelist of columns to use for features|
|cv_splits_indices|Array of integers|data_train||*Optional* List of indexes to split the data for cross validation|

<a name="preprocessing"></a>
## Data pre-processing and featurization
If you use `preprocess=True`, the following data preprocessing steps are performed automatically for you:

1. Dropping high cardinality or no variance features
    - Features with no useful information are dropped from training and validation sets. These include features with all values missing, same value across all rows or with extremely high cardinality (e.g., hashes, IDs or GUIDs).
2. Missing value imputation
    - For numerical features, missing values are imputed with average of values in the column.
    - For categorical features, missing values are imputed with most frequent value.
3. Generating additional features
    - For DateTime features: Year, Month, Day, Day of week, Day of year, Quarter, Week of the year, Hour, Minute, Second.
    - For Text features: Term frequency based on bi-grams and tri-grams, Count vectorizer.
4. Transformations and encodings
    - Numeric features with very few unique values are transformed into categorical features.

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
1. On windows, make sure that you are running automl_setup from an Anconda Prompt window rather than a regular cmd window.  You can launch the "Anaconda Prompt" window by hitting the Start button and typing "Anaconda Prompt".  If you don't see the application "Anaconda Prompt", you might not have conda or mini conda installed.  In that case, you can install it [here](https://conda.io/miniconda.html)
2. Check that you have conda 4.4.10 or later.  You can check the version with the command `conda -V`.  If you have a previous version installed, you can update it using the command: `conda update conda`.
3. Pass a new name as the first parameter to automl_setup so that it creates a new conda environment. You can view existing conda environments using `conda env list` and remove them with `conda env remove -n <environmentname>`. 

## configuration.ipynb fails
1) For local conda, make sure that you have susccessfully run automl_setup first.
2) Check that the subscription_id is correct.  You can find the subscription_id in the Azure Portal by selecting All Service and then Subscriptions. The characters "<" and ">" should not be included in the subscription_id value.  For example, `subscription_id = "12345678-90ab-1234-5678-1234567890abcd"` has the valid format.
3) Check that you have Contributor or Owner access to the Subscription.
4) Check that the region is one of the supported regions: `eastus2`, `eastus`, `westcentralus`, `southeastasia`, `westeurope`, `australiaeast`, `westus2`, `southcentralus`
5) Check that you have access to the region using the Azure Portal.

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

## Remote run: DsvmCompute.create fails 
There are several reasons why the DsvmCompute.create can fail.  The reason is usually in the error message but you have to look at the end of the error message for the detailed reason.  Some common reasons are:
1) `Compute name is invalid, it should start with a letter, be between 2 and 16 character, and only include letters (a-zA-Z), numbers (0-9) and \'-\'.`  Note that underscore is not allowed in the name.
2) `The requested VM size xxxxx is not available in the current region.`  You can select a different region or vm_size. 

## Remote run: Unable to establish SSH connection
AutoML uses the SSH protocol to communicate with remote DSVMs.  This defaults to port 22.  Possible causes for this error are:
1) The DSVM is not ready for SSH connections.  When DSVM creation completes, the DSVM might still not be ready to acceept SSH connections.  The sample notebooks have a one minute delay to allow for this.
2) Your Azure Subscription may restrict the IP address ranges that can access the DSVM on port 22.  You can check this in the Azure Portal by selecting the Virtual Machine and then clicking Networking.  The Virtual Machine name is the name that you provided in the notebook plus 10 alpha numeric characters to make the name unique.  The Inbound Port Rules define what can access the VM on specific ports.  Note that there is a priority priority order.  So, a Deny entry with a low priority number will override a Allow entry with a higher priority number.

## Remote run: setup iteration fails
This is often an issue with the `get_data` method.
1) Check that the `get_data` method is valid by running it locally.
2) Make sure that `get_data` isn't referring to any local files.  `get_data` is executed on the remote DSVM.  So, it doesn't have direct access to local data files.  Instead you can store the data files with DataStore.  See [auto-ml-remote-execution-with-datastore.ipynb](remote-execution-with-datastore/auto-ml-remote-execution-with-datastore.ipynb)
3) You can get to the error log for the setup iteration by clicking the `Click here to see the run in Azure portal` link, click `Back to Experiment`, click on the highest run number and then click on Logs.

## Remote run: disk full
AutoML creates files under /tmp/azureml_runs for each iteration that it runs.  It creates a folder with the iteration id.  For example: AutoML_9a038a18-77cc-48f1-80fb-65abdbc33abe_93.  Under this, there is a azureml-logs folder, which contains logs.  If you run too many iterations on the same DSVM, these files can fill the disk.
You can delete the files under /tmp/azureml_runs or just delete the VM and create a new one.
If your get_data downloads files, make sure the delete them or they can use disk space as well.
When using DataStore, it is good to specify an absolute path for the files so that they are downloaded just once.  If you specify a relative path, it will download a file for each iteration.

## Remote run: Iterations fail and the log contains "MemoryError"
This can be caused by insufficient memory on the DSVM.  AutoML loads all training data into memory.  So, the available memory should be more than the training data size.
If you are using a remote DSVM, memory is needed for each concurrent iteration.  The max_concurrent_iterations setting specifies the maximum concurrent iterations.  For example, if the training data size is 8Gb and max_concurrent_iterations is set to 10, the minimum memory required is at least 80Gb.
To resolve this issue, allocate a DSVM with more memory or reduce the value specified for max_concurrent_iterations.

## Remote run: Iterations show as "Not Responding" in the RunDetails widget.
This can be caused by too many concurrent iterations for a remote DSVM.  Each concurrent iteration usually takes 100% of a core when it is running.  Some iterations can use multiple cores.  So, the max_concurrent_iterations setting should always be less than the number of cores of the DSVM.
To resolve this issue, try reducing the value specified for the max_concurrent_iterations setting.