# Azure Machine Learning Datasets (preview)

Azure Machine Learning Datasets (preview) make it easier to access and work with your data. Datasets manage data in various scenarios such as model training and pipeline creation. Using the Azure Machine Learning SDK, you can access underlying storage, explore and prepare data, manage the life cycle of different Dataset definitions, and compare between Datasets used in training and in production.

## Create and Register Datasets

It's easy to create Datasets from either local files, or Azure Datastores.

```Python
from azureml.core.workspace import Workspace
from azureml.core.datastore import Datastore
from azureml.core.dataset import Dataset

datastore_name = 'your datastore name'

# get existing workspace
workspace = Workspace.from_config()

# get Datastore from the workspace
dstore = Datastore.get(workspace, datastore_name)

# create an in-memory Dataset on your local machine
dataset = Dataset.from_delimited_files(dstore.path('data/src/crime.csv'))
```

To consume Datasets across various scenarios in Azure Machine Learning service such as automated machine learning, model training and pipeline creation, you need to register the Datasets with your workspace. By doing so, you can also share and reuse the Datasets within your organization.

```Python
dataset = dataset.register(workspace = workspace,
                           name = 'dataset_crime',
                           description = 'Training data'
                           )
```

## Sampling

Sampling can be particular useful with Datasets that are too large to efficiently analyze in full. It enables data scientists to work with a manageable amount of data to build and train machine learning models. At this time, the [`sample()`](https://docs.microsoft.com//python/api/azureml-core/azureml.core.dataset(class)?view=azure-ml-py#sample-sample-strategy--arguments-) method from the Dataset class supports Top N, Simple Random, and Stratified sampling strategies.

After sampling, you can convert your sampled Dataset to pandas DataFrame for training. By using  the native [`sample()`](https://docs.microsoft.com//python/api/azureml-core/azureml.core.dataset(class)?view=azure-ml-py#sample-sample-strategy--arguments-) method from the Dataset class, you will load the sampled data on the fly instead of loading full data into memory.

### Top N sample

For Top N sampling, the first n records of your Dataset are your sample. This is helpful if you are just trying  to get an idea of what your data records look like or to see what fields are in your data.

```Python
top_n_sample_dataset = dataset.sample('top_n', {'n': 5})
top_n_sample_dataset.to_pandas_dataframe()
```

### Simple random sample

In Simple Random sampling, every member of the data population has an equal chance of being selected as a part of the sample. In the `simple_random` sample strategy, the records from your Dataset are selected based on the probability specified and returns a modified Dataset. The seed parameter is optional.

```Python
simple_random_sample_dataset = dataset.sample('simple_random', {'probability':0.3, 'seed': seed})
simple_random_sample_dataset.to_pandas_dataframe()
```

### Stratified sample

Stratified samples ensure that certain groups of a population are represented in the sample. In the `stratified` sample strategy, the population is divided into strata, or subgroups, based on similarities, and records are randomly selected from each strata according to the strata weights indicated by the `fractions` parameter.

In the following example, we group each record by the specified columns, and include said record based on the strata X weight information in `fractions`. If a strata is not specified or the record cannot be grouped, the default weight to sample is 0.

```Python
# take 50% of records with `Primary Type` as `THEFT` and 20% of records with `Primary Type` as `DECEPTIVE PRACTICE` into sample Dataset
fractions = {}
fractions[('THEFT',)] = 0.5
fractions[('DECEPTIVE PRACTICE',)] = 0.2

sample_dataset = dataset.sample('stratified', {'columns': ['Primary Type'], 'fractions': fractions, 'seed': seed})

sample_dataset.to_pandas_dataframe()
```

## Explore with summary statistics

 Detect anomalies, missing values, or error counts with the [`get_profile()`](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#get-profile-arguments-none--generate-if-not-exist-true--workspace-none--compute-target-none-) method. This function gets the profile and summary statistics of your data, which in turn helps determine the necessary data preparation operations to apply.

```Python
# get pre-calculated profile
# if there is no precalculated profile available or the precalculated profile is not up-to-date, this method will generate a new profile of the Dataset
dataset.get_profile()
```

||Type|Min|Max|Count|Missing Count|Not Missing Count|Percent missing|Error Count|Empty count|0.1% Quantile|1% Quantile|5% Quantile|25% Quantile|50% Quantile|75% Quantile|95% Quantile|99% Quantile|99.9% Quantile|Mean|Standard Deviation|Variance|Skewness|Kurtosis
-|----|---|---|-----|-------------|-----------------|---------------|-----------|-----------|-------------|-----------|-----------|------------|------------|------------|------------|------------|--------------|----|------------------|--------|--------|--------
ID|FieldType.INTEGER|1.04986e+07|1.05351e+07|10.0|0.0|10.0|0.0|0.0|0.0|1.04986e+07|1.04992e+07|1.04986e+07|1.05166e+07|1.05209e+07|1.05259e+07|1.05351e+07|1.05351e+07|1.05351e+07|1.05195e+07|12302.7|1.51358e+08|-0.495701|-1.02814
Case Number|FieldType.STRING|HZ239907|HZ278872|10.0|0.0|10.0|0.0|0.0|0.0||||||||||||||
Date|FieldType.DATE|2016-04-04 23:56:00+00:00|2016-04-15 17:00:00+00:00|10.0|0.0|10.0|0.0|0.0|0.0||||||||||||||
Block|FieldType.STRING|004XX S KILBOURN AVE|113XX S PRAIRIE AVE|10.0|0.0|10.0|0.0|0.0|0.0||||||||||||||
IUCR|FieldType.INTEGER|810|1154|10.0|0.0|10.0|0.0|0.0|0.0|810|850|810|890|1136|1153|1154|1154|1154|1058.5|137.285|18847.2|-0.785501|-1.3543
Primary Type|FieldType.STRING|DECEPTIVE PRACTICE|THEFT|10.0|0.0|10.0|0.0|0.0|0.0||||||||||||||
Description|FieldType.STRING|BOGUS CHECK|OVER $500|10.0|0.0|10.0|0.0|0.0|0.0||||||||||||||
Location Description|FieldType.STRING||SCHOOL, PUBLIC, BUILDING|10.0|0.0|10.0|0.0|0.0|1.0||||||||||||||
Arrest|FieldType.BOOLEAN|False|False|10.0|0.0|10.0|0.0|0.0|0.0||||||||||||||
Domestic|FieldType.BOOLEAN|False|False|10.0|0.0|10.0|0.0|0.0|0.0||||||||||||||
Beat|FieldType.INTEGER|531|2433|10.0|0.0|10.0|0.0|0.0|0.0|531|531|531|614|1318.5|1911|2433|2433|2433|1371.1|692.094|478994|0.105418|-1.60684
District|FieldType.INTEGER|5|24|10.0|0.0|10.0|0.0|0.0|0.0|5|5|5|6|13|19|24|24|24|13.5|6.94822|48.2778|0.0930109|-1.62325
Ward|FieldType.INTEGER|1|48|10.0|0.0|10.0|0.0|0.0|0.0|1|5|1|9|22.5|40|48|48|48|24.5|16.2635|264.5|0.173723|-1.51271
Community Area|FieldType.INTEGER|4|77|10.0|0.0|10.0|0.0|0.0|0.0|4|8.5|4|24|37.5|71|77|77|77|41.2|26.6366|709.511|0.112157|-1.73379
FBI Code|FieldType.INTEGER|6|11|10.0|0.0|10.0|0.0|0.0|0.0|6|6|6|6|11|11|11|11|11|9.4|2.36643|5.6|-0.702685|-1.59582
X Coordinate|FieldType.INTEGER|1.16309e+06|1.18336e+06|10.0|7.0|3.0|0.7|0.0|0.0|1.16309e+06|1.16309e+06|1.16309e+06|1.16401e+06|1.16678e+06|1.17921e+06|1.18336e+06|1.18336e+06|1.18336e+06|1.17108e+06|10793.5|1.165e+08|0.335126|-2.33333
Y Coordinate|FieldType.INTEGER|1.8315e+06|1.908e+06|10.0|7.0|3.0|0.7|0.0|0.0|1.8315e+06|1.8315e+06|1.8315e+06|1.83614e+06|1.85005e+06|1.89352e+06|1.908e+06|1.908e+06|1.908e+06|1.86319e+06|39905.2|1.59243e+09|0.293465|-2.33333
Year|FieldType.INTEGER|2016|2016|10.0|0.0|10.0|0.0|0.0|0.0|2016|2016|2016|2016|2016|2016|2016|2016|2016|2016|0|0|NaN|NaN
Updated On|FieldType.DATE|2016-05-11 15:48:00+00:00|2016-05-27 15:45:00+00:00|10.0|0.0|10.0|0.0|0.0|0.0||||||||||||||
Latitude|FieldType.DECIMAL|41.6928|41.9032|10.0|7.0|3.0|0.7|0.0|0.0|41.6928|41.6928|41.6928|41.7057|41.7441|41.8634|41.9032|41.9032|41.9032|41.78|0.109695|0.012033|0.292478|-2.33333
Longitude|FieldType.DECIMAL|-87.6764|-87.6043|10.0|7.0|3.0|0.7|0.0|0.0|-87.6764|-87.6764|-87.6764|-87.6734|-87.6645|-87.6194|-87.6043|-87.6043|-87.6043|-87.6484|0.0386264|0.001492|0.344429|-2.33333
Location|FieldType.STRING||(41.903206037, -87.676361925)|10.0|0.0|10.0|0.0|0.0|7.0||||||||||||||


## Training with Dataset

Now that you have registered your Dataset, you can call up the Dataset and convert it to pandas DataFrame or Spark DataFrame easily in your train.py script.

```Python
# Sample train.py script
import azureml.core
import pandas as pd
import datetime
import shutil
from azureml.core import Workspace, Datastore, Dataset, Experiment, Run
from sklearn.model_selection import train_test_split
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from sklearn.tree import DecisionTreeClassifier

run = Run.get_context()
workspace = run.experiment.workspace

# Access Dataset registered with the workspace by name
dataset_name = 'training_data'
dataset = Dataset.get(workspace=workspace, name=dataset_name)

ds_def = dataset.get_definition()
dataset_val, dataset_train = ds_def.random_split(percentage=0.3)
y_df = dataset_train.keep_columns(['HasDetections']).to_pandas_dataframe()
x_df = dataset_train.drop_columns(['HasDetections']).to_pandas_dataframe()
y_val = dataset_val.keep_columns(['HasDetections']).to_pandas_dataframe()
x_val = dataset_val.drop_columns(['HasDetections']).to_pandas_dataframe()

data = {"train": {"X": x_df, "y": y_df},
        "validation": {"X": x_val, "y": y_val}}

clf = DecisionTreeClassifier().fit(data["train"]["X"], data["train"]["y"])
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(x_df, y_df)))
print('Accuracy of Decision Tree classifier on validation set: {:.2f}'.format(clf.score(x_val, y_val)))
```

For an end-to-end tutorial, you may refer to [Dataset tutorial](datasets-tutorial.ipynb). You will learn how to:
- Explore and prepare data for training the model.
- Register the Dataset in your workspace for easy access in training.
- Take snapshots of data to ensure models can be trained with the same data every time.
- Use registered Dataset in your training script.
- Create and use multiple Dataset definitions to ensure that updates to the definition don't break existing pipelines/scripts.

 

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/work-with-data/datasets/README.png) 