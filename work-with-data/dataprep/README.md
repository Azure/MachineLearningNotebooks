# Azure Machine Learning Data Prep SDK

The Azure Machine Learning Data Prep SDK helps data scientists explore, cleanse and transform data for machine learning workflows in any Python environment.

Key benefits to the SDK:
- Cross-platform functionality. Write with a single SDK and run it on Windows, macOS, or Linux.
- Intelligent transformations powered by AI, including grouping similar values to their canonical form and deriving columns by examples without custom code.
- Capability to work with large, multiple files of different schema.
- Scalability on a single machine by streaming data during processing rather than loading into memory.
- Seamless integration with other Azure Machine Learning services. You can simply pass your prepared data file into `AutoMLConfig` object for automated machine learning training.

You will find in this repo:
- [Getting Started Tutorial](tutorials/getting-started/getting-started.ipynb) for a quick introduction to the main features of Data Prep SDK.
- [Case Study Notebooks](case-studies/new-york-taxi) that present an end-to-end data preparation tutorial where users start with small dataset, profile data with statistics summary, cleanse and perform feature engineering. All transformation steps are saved in a dataflow object. Users can easily reapply the same steps on the full dataset, and run it on Spark.
- [How-To Guide Notebooks](how-to-guides) for more in-depth sample code at feature level.

## Installation
Here are the [SDK installation steps](https://aka.ms/aml-data-prep-installation).

## Documentation 
Here is more information on how to use the new Data Prep SDK:
- [SDK overview and API reference docs](http://aka.ms/data-prep-sdk) that show different classes, methods, and function parameters for the SDK.
- [Tutorial: Prep NYC taxi data](https://docs.microsoft.com/azure/machine-learning/service/tutorial-data-prep) for regression modeling and then run automated machine learning to build the model.
- [How to load data](https://docs.microsoft.com/azure/machine-learning/service/how-to-load-data) is an overview guide on how to load data using the Data Prep SDK.
- [How to transform data](https://docs.microsoft.com/azure/machine-learning/service/how-to-transform-data) is an overview guide on how to transform data. 
- [How to write data](https://docs.microsoft.com/azure/machine-learning/service/how-to-write-data) is an overview guide on how to write data to different storage locations. 

## Support

If you have any questions or feedback, send us an email at: [askamldataprep@microsoft.com](mailto:askamldataprep@microsoft.com).

## Release Notes

### 2019-05-28 (version 1.1.4)

New features
- You can now use the following expression language functions to extract and parse datetime values into new columns.
  - `RegEx.extract_record()` extracts datetime elements into a new column.
  - `create_datetime()` creates datetime objects from separate datetime elements.
- When calling `get_profile()`, you can now see that quantile columns are labeled as (est.) to clearly indicate that the values are approximations.
- You can now use ** globbing when reading from Azure Blob Storage.
  - e.g. `dprep.read_csv(path='https://yourblob.blob.core.windows.net/yourcontainer/**/data/*.csv')`

Bug fixes
- Fixed a bug related to reading a Parquet file from a remote source (Azure Blob).

### 2019-05-08 (version 1.1.3)

New features
- Added support to read from a PostgresSQL database, either by calling `read_postgresql` or using a Datastore.
  - See examples in how-to guides:
    - [Data Ingestion notebook](https://aka.ms/aml-data-prep-ingestion-nb)
    - [Datastore notebook](https://aka.ms/aml-data-prep-datastore-nb)

Bug fixes and improvements
- Fixed issues with column type conversion:
  - Now correctly converts a boolean or numeric column to a boolean column.
  - Now does not fail when attempting to set a date column to be date type.
- Improved JoinType types and accompanying reference documentation. When joining two dataflows, you can now specify one of these types of join:
  - NONE, MATCH, INNER, UNMATCHLEFT, LEFTANTI, LEFTOUTER, UNMATCHRIGHT, RIGHTANTI, RIGHTOUTER, FULLANTI, FULL.
- Improved data type inference to recognize more date formats.

### 2019-04-17 (version 1.1.2)

Note: Data Prep Python SDK will no longer install `numpy` and `pandas` packages. See [updated installation instructions](https://aka.ms/aml-data-prep-installation).

New features
- You can now use the Pivot transform.
  - How-to guide: [Pivot notebook](https://aka.ms/aml-data-prep-pivot-nb)
- You can now use regular expressions in native functions.
  - Examples:
    - `dflow.filter(dprep.RegEx('pattern').is_match(dflow['column_name']))`
    - `dflow.assert_value('column_name', dprep.RegEx('pattern').is_match(dprep.value))`
- You can now use `to_upper` and `to_lower` functions in expression language.
- You can now see the number of unique values of each column in a data profile.
- For some of the commonly used reader steps, you can now pass in the `infer_column_types` argument. If it is set to `True`, Data Prep will attempt to detect and automatically convert column types.
  - `inference_arguments` is now deprecated.
- You can now call `Dataflow.shape`.

Bug fixes and improvements
- `keep_columns` now accepts an additional optional argument `validate_column_exists`, which checks if the result of `keep_columns` will contain any columns.
- All reader steps (which read from a file) now accept an additional optional argument `verify_exists`.
- Improved performance of reading from pandas dataframe and getting data profiles.
- Fixed a bug where slicing a single step from a Dataflow failed with a single index.

### 2019-04-08 (version 1.1.1)

New features
- You can read multiple Datastore/DataPath/DataReference sources using read_* transforms.
- You can perform the following operations on columns to create a new column: division, floor, modulo, power, length.
- Data Prep is now part of the Azure ML diagnostics suite and will log diagnostic information by default.
  - To turn this off, set this environment variable to true: DISABLE_DPREP_LOGGER

Bug fixes and improvements
- Improved code documentation for commonly used classes and functions.
- Fixed a bug in auto_read_file that failed to read Excel files.
- Added option to overwrite the folder in read_pandas_dataframe.
- Improved performance of dotnetcore2 dependency installation, and added support for Fedora 27/28 and Ubuntu 1804.
- Improved the performance of reading from Azure Blobs.
- Column type detection now supports columns of type Long.
- Fixed a bug where some date values were being displayed as timestamps instead of Python datetime objects.
- Fixed a bug where some type counts were being displayed as doubles instead of integers.

### 2019-03-25 (version 1.1.0)

Breaking changes
- The concept of the Data Prep Package has been deprecated and is no longer supported. Instead of persisting multiple Dataflows in one Package, you can persist Dataflows individually.
  - How-to guide: [Opening and Saving Dataflows notebook](https://aka.ms/aml-data-prep-open-save-dataflows-nb)

New features
- Data Prep can now recognize columns that match a particular Semantic Type, and split accordingly. The STypes currently supported include: email address, geographic coordinates (latitude & longitude), IPv4 and IPv6 addresses, US phone number, and US zip code.
  - How-to guide: [Semantic Types notebook](https://aka.ms/aml-data-prep-semantic-types-nb)
- Data Prep now supports the following operations to generate a resultant column from two numeric columns: subtract, multiply, divide, and modulo.
- You can call `verify_has_data()` on a Dataflow to check whether the Dataflow would produce records if executed.

Bug fixes and improvements
- You can now specify the number of bins to use in a histogram for numeric column profiles.
- The `read_pandas_dataframe` transform now requires the DataFrame to have string- or byte- typed column names.
- Fixed a bug in the `fill_nulls` transform, where values were not correctly filled in if the column was missing.

### 2019-03-11 (version 1.0.17)

New features
- Now supports adding two numeric columns to generate a resultant column using the expression language.

Bug fixes and improvements
- Improved the documentation and parameter checking for random_split.

### 2019-02-27 (version 1.0.16)

Bug fix
- Fixed a Service Principal authentication issue that was caused by an API change.

### 2019-02-25 (version 1.0.15)

New features
- Data Prep now supports writing file streams from a dataflow. Also provides the ability to manipulate the file stream names to create new file names.
  - How-to guide: [Working With File Streams notebook](https://aka.ms/aml-data-prep-file-stream-nb)
 
Bug fixes and improvements
- Improved performance of t-Digest on large data sets.
- Data Prep now supports reading data from a DataPath.
- One hot encoding now works on boolean and numeric columns.
- Other miscellaneous bug fixes.

### 2019-02-11 (version 1.0.12)

New features
- Data Prep now supports reading from an Azure SQL database using Datastore.
 
Changes
- Significantly improved the memory performance of certain operations on large data.
- `read_pandas_dataframe()` now requires `temp_folder` to be specified.
- The `name` property on `ColumnProfile` has been deprecated - use `column_name` instead.

### 2019-01-28 (version 1.0.8)

Bug fixes
- Significantly improved the performance of getting data profiles.
- Fixed minor bugs related to error reporting.

### 2019-01-14 (version 1.0.7)

New features
- Datastore improvements (documented in [Datastore how-to-guide](https://aka.ms/aml-data-prep-datastore-nb))
  - Added ability to read from and write to Azure File Share and ADLS Datastores in scale-up.
  - When using Datastores, Data Prep now supports using service principal authentication instead of interactive authentication.
  - Added support for wasb and wasbs urls.

### 2019-01-09 (version 1.0.6)

Bug fixes
- Fixed bug with reading from public readable Azure Blob containers on Spark.

### 2018-12-19 (version 1.0.4)

New features
- `to_bool` function now allows mismatched values to be converted to Error values. This is the new default mismatch behavior for `to_bool` and `set_column_types`, whereas the previous default behavior was to convert mismatched values to False.
- When calling `to_pandas_dataframe`, there is a new option to interpret null/missing values in numeric columns as NaN.
- Added ability to check the return type of some expressions to ensure type consistency and fail early.
- You can now call `parse_json` to parse values in a column as JSON objects and expand them into multiple columns.

Bug fixes
- Fixed a bug that crashed `set_column_types` in Python 3.5.2.
- Fixed a bug that crashed when connecting to Datastore using an AML image.

### 2018-12-07 (version 0.5.3)

Fixed missing dependency issue for .NET Core2 on Ubuntu 16.

### 2018-12-03 (version 0.5.2)

Breaking changes
- `SummaryFunction.N` was renamed to `SummaryFunction.Count`.
  
Bug fixes
- Use latest AML Run Token when reading from and writing to datastores on remote runs. Previously, if the AML Run Token is updated in Python, the Data Prep runtime will not be updated with the updated AML Run Token.
- Additional clearer error messages
- to_spark_dataframe() will no longer crash when Spark uses Kryo serialization
- Value Count Inspector can now show more than 1000 unique values
- Random Split no longer fails if the original Dataflow doesn’t have a name  

### 2018-11-19 (version 0.5.0)

New features
- Created a new DataPrep CLI to execute DataPrep packages and view the data profile for a dataset or dataflow
- Redesigned SetColumnType API to improve usability
- Renamed smart_read_file to auto_read_file
- Now includes skew and kurtosis in the Data Profile
- Can sample with stratified sampling
- Can read from zip files that contain CSV files
- Can split datasets row-wise with Random Split (e.g. into test-train sets)
- Can get all the column data types from a dataflow or a data profile by calling .dtypes
- Can get the row count from a dataflow or a data profile by calling .row_count

Bug fixes
- Fixed long to double conversion 
- Fixed assert after any add column 
- Fixed an issue with FuzzyGrouping, where it would not detect groups in some cases
- Fixed sort function to respect multi-column sort order
- Fixed and/or expressions to be similar to how Pandas handles them
- Fixed reading from dbfs path.
- Made error messages more understandable 
- Now no longer fails when reading on remote compute target using AML token
- Now no longer fails on Linux DSVM
- Now no longer crashes when non-string values are in string predicates
- Now handles assertion errors when Dataflow should fail correctly
- Now supports dbutils mounted storage locations on Azure Databricks

### 2018-11-05 (version 0.4.0)

New features
- Type Count added to Data Profile
- Value Count and Histogram is now available
- More percentiles in Data Profile
- The Median is available in Summarize
- Python 3.7 is now supported
- When you save a dataflow that contains datastores to a Data Prep package, the datastore information will be persisted as part of the Data Prep package
- Writing to datastore is now supported
 
Bug fixes
- 64bit unsigned integer overflows are now handled properly on Linux 
- Fixed incorrect text label for plain text files in smart_read
- String column type now shows up in metrics view
- Type count now is fixed to show ValueKinds mapped to single FieldType instead of individual ones
- Write_to_csv no longer fails when path is provided as a string
- When using Replace, leaving “find” blank will no longer fail

## Datasets License Information

IMPORTANT: Please read the notice and find out more about this NYC Taxi and Limousine Commission dataset here: http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml 

IMPORTANT: Please read the notice and find out more about this Chicago Police Department dataset here: https://catalog.data.gov/dataset/crimes-2001-to-present-398a4 

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/work-with-data/dataprep/README.png) 
