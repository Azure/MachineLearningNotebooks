# Table of Contents
1. [Introduction](#introduction)
1. [Setup using Azure Data Studio](#azuredatastudiosetup)
1. [Energy demand example using Azure Data Studio](#azuredatastudioenergydemand)
1. [Set using SQL Server Management Studio for SQL Server 2017 on Windows](#ssms2017)
1. [Set using SQL Server Management Studio for SQL Server 2019 on Linux](#ssms2019)
1. [Energy demand example using SQL Server Management Studio](#ssmsenergydemand)


<a name="introduction"></a>
# Introduction
SQL Server 2017 or 2019 can call Azure ML automated machine learning to create models trained on data from SQL Server.
This uses the sp_execute_external_script stored procedure, which can call Python scripts.
SQL Server 2017 and SQL Server 2019 can both run on Windows or Linux.
However, this integration is not available for SQL Server 2017 on Linux. 

This folder shows how to setup the integration and has a sample that uses the integration to train and predict based on an energy demand dataset.

This integration is part of SQL Server and so can be used from any SQL client. 
These instructions show using it from Azure Data Studio or SQL Server Managment Studio.

<a name="azuredatastudiosetup"></a>
## Setup using Azure Data Studio

These step show setting up the integration using Azure Data Studio.

1. If you don't already have SQL Server, you can install it from [https://www.microsoft.com/en-us/sql-server/sql-server-downloads](https://www.microsoft.com/en-us/sql-server/sql-server-downloads)
1. Install Azure Data Studio from [https://docs.microsoft.com/en-us/sql/azure-data-studio/download?view=sql-server-2017](https://docs.microsoft.com/en-us/sql/azure-data-studio/download?view=sql-server-2017)
1. Start Azure Data Studio and connect to SQL Server. [https://docs.microsoft.com/en-us/sql/azure-data-studio/sql-notebooks?view=sql-server-2017](https://docs.microsoft.com/en-us/sql/azure-data-studio/sql-notebooks?view=sql-server-2017)
1. Create a database named "automl".
1. Open the notebook how-to-use-azureml\automated-machine-learning\sql-server\setup\auto-ml-sql-setup.ipynb and follow the instructions in it.

 <a name="azuredatastudioenergydemand"></a>
## Energy demand example using Azure Data Studio

Once you have completed the setup, you can try the energy demand sample in the notebook energy-demand\auto-ml-sql-energy-demand.ipynb.
This has cells to train a model, predict based on the model and show metrics for each pipeline run in training the model.

<a name="ssms2017"></a>
## Setup using SQL Server Management Studio for SQL Server 2017 on Windows

These instruction setup the integration for SQL Server 2017 on Windows.

1. If you don't already have SQL Server, you can install it from [https://www.microsoft.com/en-us/sql-server/sql-server-downloads](https://www.microsoft.com/en-us/sql-server/sql-server-downloads)
2. Enable external scripts with the following commands: 
```sh
   sp_configure 'external scripts enabled',1 
   reconfigure with override
```
3. Stop SQL Server. 
4. Install the automated machine learning libraries using the following commands from Administrator command prompt (If you are using a non-default SQL Server instance name, replace MSSQLSERVER in the second command with the instance name)
```sh
   cd "C:\Program Files\Microsoft SQL Server"
   cd "MSSQL14.MSSQLSERVER\PYTHON_SERVICES"
   python.exe -m pip install azureml-sdk[automl]
   python.exe -m pip install --upgrade numpy
   python.exe -m pip install --upgrade sklearn
```
5. Start SQL Server and the service "SQL Server Launchpad service". 
6. In Windows Firewall, click on advanced settings and in Outbound Rules, disable "Block network access for R local user accounts in SQL Server instance xxxx". 
7. Execute the files in the setup folder in SQL Server Management Studio: aml_model.sql, aml_connection.sql, AutoMLGetMetrics.sql, AutoMLPredict.sql and AutoMLTrain.sql 
8. Create an Azure Machine Learning Workspace.  You can use the instructions at: [https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace ](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace)
9. Create a config.json file file using the subscription id, resource group name and workspace name that you used to create the workspace.  The file is described at: [https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#workspace)
10. Create an Azure service principal.  You can do this with the commands: 
```sh
   az login 
   az account set --subscription subscriptionid 
   az ad sp create-for-rbac --name principlename --password password 
```
11. Insert the values \<tenant\>, \<AppId\> and \<password\> returned by create-for-rbac above into the aml_connection table.  Set \<path\> as the absolute path to your config.json file. Set the name to “Default”. 
 
<a name="ssms2019"></a>
## Setup using SQL Server Management Studio for SQL Server 2019 on Linux
1. Install SQL Server 2019 from: [https://www.microsoft.com/en-us/sql-server/sql-server-downloads](https://www.microsoft.com/en-us/sql-server/sql-server-downloads)
2. Install machine learning support from: [https://docs.microsoft.com/en-us/sql/linux/sql-server-linux-setup-machine-learning?view=sqlallproducts-allversions#ubuntu](https://docs.microsoft.com/en-us/sql/linux/sql-server-linux-setup-machine-learning?view=sqlallproducts-allversions#ubuntu)
3. Then install SQL Server management Studio from [https://docs.microsoft.com/en-us/sql/ssms/download-sql-server-management-studio-ssms?view=sql-server-2017](https://docs.microsoft.com/en-us/sql/ssms/download-sql-server-management-studio-ssms?view=sql-server-2017)
4. Enable external scripts with the following commands: 
```sh
   sp_configure 'external scripts enabled',1 
   reconfigure with override 
```
5. Stop SQL Server. 
6. Install the automated machine learning libraries using the following commands from Administrator command (If you are using a non-default SQL Server instance name, replace MSSQLSERVER in the second command with the instance name): 
```sh
   sudo /opt/mssql/mlservices/bin/python/python -m pip install azureml-sdk[automl] 
   sudo /opt/mssql/mlservices/bin/python/python -m pip install --upgrade numpy 
   sudo /opt/mssql/mlservices/bin/python/python -m pip install --upgrade sklearn
```
7. Start SQL Server. 
8. Execute the files aml_model.sql, aml_connection.sql, AutoMLGetMetrics.sql, AutoMLPredict.sql, AutoMLForecast.sql and AutoMLTrain.sql in SQL Server Management Studio. 
9. Create an Azure Machine Learning Workspace.  You can use the instructions at: [https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace)
10. Create a config.json file file using the subscription id, resource group name and workspace name that you use to create the workspace.  The file is described at: [https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#workspace)
11. Create an Azure service principal.  You can do this with the commands: 
```sh
   az login 
   az account set --subscription subscriptionid 
   az ad sp create-for-rbac --name principlename --password password 
``` 
12. Insert the values \<tenant\>, \<AppId\> and \<password\> returned by create-for-rbac above into the aml_connection table.  Set \<path\> as the absolute path to your config.json file. Set the name to “Default”. 
 
<a name="ssmsenergydemand"></a>
## Energy demand example using SQL Server Management Studio

Once you have completed the setup, you can try the energy demand sample queries.
First you need to load the sample data in the database.
1. In SQL Server Management Studio, you can right-click the database, select Tasks, then Import Flat file. 
1. Select the file MachineLearningNotebooks\notebooks\how-to-use-azureml\automated-machine-learning\forecasting-energy-demand\nyc_energy.csv. 
1. When you get to the column definition page, allow nulls for all columns. 

You can then run the queries in the energy-demand folder:
* TrainEnergyDemand.sql runs AutoML, trains multiple models on data and selects the best model.
* ForecastEnergyDemand.sql forecasts based on the most recent training run.
* GetMetrics.sql returns all the metrics for each model in the most recent training run.
