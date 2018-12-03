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
Start an **Anaconda Prompt** window, cd to the **automl** folder where the sample notebooks were extracted and then run:
```
automl_setup
```
## Mac
Install "Command line developer tools" if it is not already installed (you can use the command: `xcode-select --install`).

Start a Terminal windows, cd to the **automl** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_mac.sh
```

## Linux
cd to the **automl** folder where the sample notebooks were extracted and then run:

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
    - Dataset: [Burning Man 2016 dataset](https://innovate.burningman.org/datasets-page/)
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
    - Dataset: scikit learn's [digit dataset](https://innovate.burningman.org/datasets-page/)
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

- [auto-ml-timeseries.ipynb](timeseries/auto-ml-timeseries.ipynb)
    - Dataset: NYC energy demanding data
    - Example of using AutoML for timeseries data training

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
## Iterations fail and the log contains "MemoryError"
This can be caused by insufficient memory on the DSVM.  AutoML loads all training data into memory.  So, the available memory should be more than the training data size.
If you are using a remote DSVM, memory is needed for each concurrent iteration.  The max_concurrent_iterations setting specifies the maximum concurrent iterations.  For example, if the training data size is 8Gb and max_concurrent_iterations is set to 10, the minimum memory required is at least 80Gb.
To resolve this issue, allocate a DSVM with more memory or reduce the value specified for max_concurrent_iterations.

## Iterations show as "Not Responding" in the RunDetails widget.
This can be caused by too many concurrent iterations for a remote DSVM.  Each concurrent iteration usually takes 100% of a core when it is running.  Some iterations can use multiple cores.  So, the max_concurrent_iterations setting should always be less than the number of cores of the DSVM.
To resolve this issue, try reducing the value specified for the max_concurrent_iterations setting.