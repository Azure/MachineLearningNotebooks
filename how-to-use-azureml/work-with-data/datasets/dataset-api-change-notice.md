# Dataset API change notice

## Why are Dataset API changes essential?

The existing Dataset class only supports data in tabular format. In order to support binary data and address a wider range of machine learning scenarios including deep learning, we will introduce Dataset types. Datasets are categorized into various types based on how users consume them in training. List of Dataset types:
- **TabularDataset**: Represents data in a tabular format by parsing the provided file or list of files. TabularDataset can be created from csv, tsv, parquet files, SQL query results etc. For the complete list, please visit our [documentation](https://aka.ms/tabulardataset-api-reference). It provides you with the ability to materialize the data into a pandas DataFrame.
- (upcoming) **FileDataset**: References single or multiple files in your datastores or public urls. The files can be of any format. FileDataset provides you with the ability to download or mount the files to your compute.
- (upcoming) **LabeledDataset**: Represents labeled data that are produced by Azure Machine Learning Labeling service. LabaledDataset provides you with the ability to materialize the data into formats like [COCO](http://cocodataset.org/#homeo) or [TFRecord](https://www.tensorflow.org/tutorials/load_data/tf_records) on your compute.
- (upcoming) **TimeSeriesDataset**: An extension of TabularDataset that allows for specification of a time column and filtering the Dataset by time.

In order to transit from the current Dataset design to typed Dataset, we will deprecate a series of methods on the Dataset class and launch the FileDataset and TabularDataset classes.

## Which methods on Dataset class will be deprecated in upcoming releases?
Methods to be deprecated|Replacement in the new version|
----|--------
[Dataset.get()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#get-workspace--name-none--id-none-)|`Dataset.get_by_name()`
[Dataset.from_pandas_dataframe()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#from-pandas-dataframe-dataframe--path-none--in-memory-false-)|Creating a Dataset from in-memory DataFrame or local files will cause errors in training on remote compute. Therefore, the new Dataset design will only support creating Datasets from paths in datastores or public web urls. If you are using pandas, you can write the DataFrame into a parquet file, upload it to the cloud, and create a TabularDataset referencing the parquet file using `Dataset.Tabular.from_parquet_files()`
[Dataset.from_delimited_files()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#from-delimited-files-path--separator------header--promoteheadersbehavior-all-files-have-same-headers--3---encoding--fileencoding-utf8--0---quoting-false--infer-column-types-true--skip-rows-0--skip-mode--skiplinesbehavior-no-rows--0---comment-none--include-path-false--archive-options-none--partition-format-none-)|`Dataset.Tabular.from_delimited_files()`
[Dataset.auto_read_files()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#auto-read-files-path--include-path-false--partition-format-none-)|`auto_read_files` does not always produce results that match with users' expectation. To avoid confusion, this method is not introduced with TabularDataset for now. Please use `Dataset.Tabular.from_parquet_files()` or `Dataset.Tabular.from_delimited_files()` depending on your file format.
[Dataset.from_parquet_files()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#from-parquet-files-path--include-path-false--partition-format-none-)|`Dataset.Tabular.from_parquet_files()`
[Dataset.from_sql_query()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#from-sql-query-data-source--query-)|`Dataset.Tabular.from_sql_query()`
[Dataset.from_excel_files()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#from-excel-files-path--sheet-name-none--use-column-headers-false--skip-rows-0--include-path-false--infer-column-types-true--partition-format-none-)|We will support creating a TabularDataset from Excel files in a future release.
[Dataset.from_json_files()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#from-json-files-path--encoding--fileencoding-utf8--0---flatten-nested-arrays-false--include-path-false--partition-format-none-)| We will support creating a TabularDataset from json files in a future release.
[Dataset.to_pandas_dataframe()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#to-pandas-dataframe--)|`TabularDataset.to_pandas_dataframe()`
[Dataset.to_spark_dataframe()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#to-spark-dataframe--)|`TabularDataset.to_spark_dataframe()`
[Dataset.head(3)](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#head-count-)|`TabularDataset.take(3).to_pandas_dataframe()`
[Dataset.sample()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#sample-sample-strategy--arguments-)|`TabularDataset.take_sample()`
[Dataset.from_binary_files()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py#from-binary-files-path-)|`Dataset.File.from_files()`


## Why should I use the new Dataset API if I'm only dealing with tabular data?
The current Dataset will be kept around for backward compatibility, but we strongly encourage you to move to TabularDataset for the new capabilities listed below: 

- You are able to use TabularDatasets as automated ML input. [Learn How](https://aka.ms/automl-dataset)
- You are able to version the new typed Datasets. [Learn How](https://aka.ms/azureml/howto/createdatasets)
- You will be able to use the new typed Datasets as ScriptRun, Estimator, HyperDrive input.
- You will be able to use the new typed Datasets in Azure Machine Learning Pipelines.
- You will be able to track the lineage of new typed Datasets for model reproducibility.


## How to migrate registered Datasets to new typed Datasets?
If you have registered Datasets created using the old API, you can easily migrate these old Datasets to the new typed Datasets using the following code.
```Python
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset

# get existing workspace
workspace = Workspace.from_config()
# This method will convert old Dataset without type to a TabularDataset object automatically.
new_ds = Dataset.get_by_name(workapce, 'old_ds_name')

# register the new typed Dataset with the workspace
new_ds.register(workspace, 'new_ds_name')
```

## How to provide feedback?
If you have any feedback about our product, or if there is any missing capability that is essential for you to use new Dataset API, please email us at [AskAzureMLData@microsoft.com](mailto:AskAzureMLData@microsoft.com).