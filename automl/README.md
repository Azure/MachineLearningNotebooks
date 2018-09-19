# Table of Contents
1. [Auto ML Introduction](#introduction)
2. [Running samples in a Local Conda environment](#localconda)
3. [Auto ML SDK Sample Notebooks](#samples)
4. [Documentation](#documentation)
5. [Running using python command](#pythoncommand)
6. [Troubleshooting](#troubleshooting)

# Auto ML Introduction <a name="introduction"></a>
AutoML builds high quality Machine Learning model for you by automating model selection and hyper parameter selection for you. Bring a labelled dataset that you want to build a model for, AutoML will give you a high quality machine learning model that you can use for predictions.

If you are new to Data Science, AutoML will help you get jumpstarted by simplifying machine learning model building. It abstracts you from needing to perform model selection, hyper parameter selection and in one step creates a high quality trained model for you to use.

If you are an experienced data scientist, AutoML will help increase your productivity by intelligently performing the model selection, hyper parameter selection for your training and generates high quality models much quicker than manually specifying several combinations of the parameters and running training jobs. AutoML provides visibility and access to all the training jobs and the performance characteristics of the models and help you further tune the pipeline if you desire.

# Running samples in a Local Conda environment <a name="localconda"></a>

It is best if you create a new conda environment locally to try this SDK, so it doesn't mess up with your existing Python environment. 

### 1. Install mini-conda from [here](https://conda.io/miniconda.html), choose Python 3.7 or higher. 
- **Note**: if you already have conda installed, you can keep using it but it must be version 5.2 or later.  If you have an previous version installed, you can update it using the command: conda update conda.
There's no need to install mini-conda specifically.

### 2. Dowloading the sample notebooks
- Download the sample notebooks from [GitHub](https://github.com/Azure/MachineLearningNotebooks) as zip and extract the contents to a local directory.  The AutoML sample notebooks are in the "automl" folder.

### 3. Setup a new conda environment
The automl_setup script creates a new conda environment, installs the necessary packages, configures the widget and starts jupyter notebook.
It takes the conda environment name as an optional parameter.  The default conda environment name is azure_automl.  The exact command depends on the operating system.  It can take about 30 minutes to execute.
## Windows
Start a conda command windows, cd to the "automl" folder where the sample notebooks were extracted and then run: automl_setup
## Mac
Install "Command line developer tools" if it is not already installed (you can use the command: xcode-select --install).
Start a Terminal windows, cd to the "automl" folder where the sample notebooks were extracted and then run: bash automl_setup_mac.sh
## Linux
cd to the "automl" folder where the sample notebooks were extracted and then run: automl_setup_linux.sh

### 4. Running configuration.ipynb
- Before running any samples you would need to run the configuration notebook. Click on 00.configuration.ipynb notebook
- Please make sure you use the Python [conda env:azure_automl] kernel when running this notebook.
- Execute the cells in the notebook to Register Machine Learning Services Resource Provider and create a workspace. (*instructions in notebook*)

### 5. Running Samples
- Please make sure you use the Python [conda env:azure_automl] kernel when trying the sample Notebooks.
- Follow the instructions in the individual notebooks to explore various features in AutoML

# Auto ML SDK Sample Notebooks <a name="samples"></a>
- [00.configuration.ipynb](00.configuration.ipynb)
    - Register Machine Learning Services Resource Provider
    - Create new Azure ML Workspace
    - Save Workspace configuration file

- [01.auto-ml-classification.ipynb](01.auto-ml-classification.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Simple example of using Auto ML for classification 
    - Uses local compute for training

- [02.auto-ml-regression.ipynb](02.auto-ml-regression.ipynb)
    - Dataset: scikit learn's [diabetes dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
    - Simple example of using Auto ML for regression 
    - Uses local compute for training

- [03.auto-ml-remote-execution.ipynb](03.auto-ml-remote-execution.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Example of using Auto ML for classification using a remote linux DSVM for training
    - Parallel execution of iterations
    - Async tracking of progress
    - Cancelling individual iterations or entire run
    - Retrieving models for any iteration or logged metric
    - Specify automl settings as kwargs

- [03b.auto-ml-remote-batchai.ipynb](03b.auto-ml-remote-batchai.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Example of using Auto ML for classification using a remote Batch AI compute for training
    - Parallel execution of iterations
    - Async tracking of progress
    - Cancelling individual iterations or entire run
    - Retrieving models for any iteration or logged metric
    - Specify automl settings as kwargs

- [04.auto-ml-remote-execution-text-data-blob-store.ipynb](04.auto-ml-remote-execution-text-data-blob-store.ipynb)
    - Dataset: [Burning Man 2016 dataset](https://innovate.burningman.org/datasets-page/)
    - handling text data with preprocess flag
    - Reading data from a blob store for remote executions
    - using pandas dataframes for reading data    

- [05.auto-ml-missing-data-blacklist-early-termination.ipynb](05.auto-ml-missing-data-blacklist-early-termination.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Blacklist certain pipelines
    - Specify a target metrics to indicate stopping criteria
    - Handling Missing Data in the input

- [06.auto-ml-sparse-data-custom-cv-split.ipynb](06.auto-ml-sparse-data-custom-cv-split.ipynb)
    - Dataset: Scikit learn's [20newsgroup](http://scikit-learn.org/stable/datasets/twenty_newsgroups.html) 
    - Handle sparse datasets
    - Specify custom train and validation set

- [07.auto-ml-exploring-previous-runs.ipynb](07.auto-ml-exploring-previous-runs)
    - List all projects for the workspace
    - List all AutoML Runs for a given project
    - Get details for a AutoML Run. (Automl settings, run widget & all metrics)
    - Downlaod fitted pipeline for any iteration

- [08.auto-ml-remote-execution-with-text-file-on-DSVM](08.auto-ml-remote-execution-with-text-file-on-DSVM.ipynb)
    - Dataset: scikit learn's [digit dataset](https://innovate.burningman.org/datasets-page/)
    - Download the data and store it in the DSVM to improve performance.

- [09.auto-ml-classification-with-deployment.ipynb](09.auto-ml-classification-with-deployment.ipynb)
    - Dataset: scikit learn's [digit dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    - Simple example of using Auto ML for classification 
    - Registering the model
    - Creating Image and creating aci service
    - Testing the aci service

- [10.auto-ml-multi-output-example.ipynb](10.auto-ml-multi-output-example.ipynb)
    - Dataset: scikit learn's random example using multi-output pipeline(http://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_regression_multioutput.html#sphx-glr-auto-examples-ensemble-plot-random-forest-regression-multioutput-py)
    - Simple example of using Auto ML for multi output regression
    - Handle both the dense and sparse metrix

- [11.auto-ml-sample-weight.ipynb](11.auto-ml-sample-weight.ipynb)
    - How to specifying sample_weight
    - The difference that it makes to test results

- [12.auto-ml-retrieve-the-training-sdk-versions.ipynb](12.auto-ml-retrieve-the-training-sdk-versions.ipynb)
    - How to get current and training env SDK versions

- [13.auto-ml-dataprep.ipynb](13.auto-ml-dataprep.ipynb)
    - Using DataPrep for reading data

# Documentation <a name="documentation"></a>
## Table of Contents
1. [Auto ML Settings ](#automlsettings)
2. [Cross validation split options](#cvsplits)
3. [Get Data Syntax](#getdata)

## Auto ML Settings <a name="automlsettings"></a>
|Property|Description|Default|
|-|-|-|
|**primary_metric**|This is the metric that you want to optimize.<br><br> Classification supports the following primary metrics <br><i>accuracy</i><br><i>AUC_weighted</i><br><i>balanced_accuracy</i><br><i>average_precision_score_weighted</i><br><i>precision_score_weighted</i><br><br> Regression supports the following primary metrics <br><i>spearman_correlation</i><br><i>normalized_root_mean_squared_error</i><br><i>r2_score</i><br><i>normalized_mean_absolute_error</i><br><i>normalized_root_mean_squared_log_error</i>| Classification: accuracy <br><br> Regression: spearman_correlation
|**max_time_sec**|Time limit in seconds for each iterations|None|
|**iterations**|Number of iterations. In each iteration trains the data with a specific pipeline.  To get the best result, use at least 100. |25|
|**n_cross_validations**|Number of cross validation splits|None|
|**validation_size**|Size of validation set as percentage of all training samples|None|
|**concurrent_iterations**|Max number of iterations that would be executed in parallel|1|
|**preprocess**|*True/False* <br>Setting this to *True* enables preprocessing <br>on the input to handle *missing data*, and perform some common *feature extraction*<br>*Note: If input data is Sparse you cannot use preprocess=True*|False|
|**max_cores_per_iteration**| Indicates how many cores on the compute target would be used to train a single pipeline.<br> You can set it to *-1* to use all cores|1|
|**exit_score**|*double* value indicating the target for *primary_metric*. <br> Once the target is surpassed the run terminates|None|
|**blacklist_algos**|*Array* of *strings* indicating pipelines to ignore for Auto ML.<br><br> Allowed values for **Classification**<br><i>LogisticRegression</i><br><i>SGDClassifierWrapper</i><br><i>NBWrapper</i><br><i>BernoulliNB</i><br><i>SVCWrapper</i><br><i>LinearSVMWrapper</i><br><i>KNeighborsClassifier</i><br><i>DecisionTreeClassifier</i><br><i>RandomForestClassifier</i><br><i>ExtraTreesClassifier</i><br><i>gradient boosting</i><br><i>LightGBMClassifier</i><br><br>Allowed values for **Regression**<br><i>ElasticNet</i><br><i>GradientBoostingRegressor</i><br><i>DecisionTreeRegressor</i><br><i>KNeighborsRegressor</i><br><i>LassoLars</i><br><i>SGDRegressor</i><br><i>RandomForestRegressor</i><br><i>ExtraTreesRegressor</i>|None|

## Cross validation split options <a name="cvsplits"></a>
### K-Folds Cross Validation
Use *n_cross_validations* setting to specify the number of cross validations. The training data set will be randomly split into *n_cross_validations* folds of equal size. During each cross validation round, one of the folds will be used for validation of the model trained on the remaining folds. This process repeats for *n_cross_validations* rounds until each fold is used once as validation set. Finally, the average scores accross all *n_cross_validations* rounds will be reported, and the corresponding model will be retrained on the whole training data set.

### Monte Carlo Cross Validation (a.k.a. Repeated Random Sub-Sampling)
Use *validation_size* to specify the percentage of the training data set that should be used for validation, and use *n_cross_validations* to specify the number of cross validations. During each cross validation round, a subset of size *validation_size* will be randomly selected for validation of the model trained on the remaining data. Finally, the average scores accross all *n_cross_validations* rounds will be reported, and the corresponding model will be retrained on the whole training data set.

### Custom train and validation set
You can specify seperate train and validation set either through the get_data() or directly to the fit method.

## get_data() syntax <a name="getdata"></a>
The *get_data()* function can be used to return a dictionary with these values:

|Key|Type|Dependency|Mutually Exclusive with|Description|
|:-|:-|:-|:-|:-|
|X|Pandas Dataframe or Numpy Array|y|data_train, label, columns|All features to train with|
|y|Pandas Dataframe or Numpy Array|X|label|Label data to train with.  For classification, this should be an array of integers. |
|X_valid|Pandas Dataframe or Numpy Array|X, y, y_valid|data_train, label|*Optional* All features to validate with.  If this is not specified, X is split between train and validate|
|y_valid|Pandas Dataframe or Numpy Array|X, y, X_valid|data_train, label|*Optional* The label data to validate with.  If this is not specified, y is split between train and validate|
|sample_weight|Pandas Dataframe or Numpy Array|y|data_train, label, columns|*Optional*A weight value for each label. Higher values indicate that the sample is more important.|
|sample_weight_valid|Pandas Dataframe or Numpy Array|y_valid|data_train, label, columns|*Optional*A weight value for each validation label. Higher values indicate that the sample is more important.  If this is not specified, sample_weight is split between train and validate|
|data_train|Pandas Dataframe|label|X, y, X_valid, y_valid|All data (features+label) to train with|
|label|string|data_train|X, y, X_valid, y_valid|Which column in data_train represents the label|
|columns|Array of strings|data_train||*Optional* Whitelist of columns to use for features|
|cv_splits_indices|Array of integers|data_train||*Optional* List of indexes to split the data for cross validation|

# Running using python command <a name="pythoncommand"></a>
Jupyter notebook provides a File / Download as / Python (.py) option for saving the notebook as a Python file.
You can then run this file using the python command.
However, on Windows the file needs to be modified before it can be run.
The following condition must be added to the main code in the file:

    if __name__ == "__main__":

The main code of the file must be indented so that it is under this condition.

# Troubleshooting <a name="troubleshooting"></a>
## Iterations fail and the log contains "MemoryError"
This can be caused by insufficient memory on the DSVM.  AutoML loads all training data into memory.  So, the available memory should be more than the training data size.
If you are using a remote DSVM, memory is needed for each concurrent iteration.  The concurrent_iterations setting specifies the maximum concurrent iterations.  For example, if the training data size is 8Gb and concurrent_iterations is set to 10, the minimum memory required is at least 80Gb.
To resolve this issue, allocate a DSVM with more memory or reduce the value specified for concurrent_iterations.

## Iterations show as "Not Responding" in the RunDetails widget.
This can be caused by too many concurrent iterations for a remote DSVM.  Each concurrent iteration usually takes 100% of a core when it is running.  Some iterations can use multiple cores.  So, the concurrent_iterations setting should always be less than the number of cores of the DSVM.
To resolve this issue, try reducing the value specified for the concurrent_iterations setting.

## Workspace.create gives the error "The resource type could not be found in the namespace 'Microsoft.MachineLearningServices' for api version '2018-03-01-preview'."
This can indicate that the Azure Subscription has not been whitelisted for AutoML.
