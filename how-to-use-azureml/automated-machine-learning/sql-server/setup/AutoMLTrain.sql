-- This stored procedure uses automated machine learning to train several models
-- and returns the best model.
--
-- The result set has several columns:
--   best_run - iteration ID for the best model
--   experiment_name - experiment name pass in with the @experiment_name parameter
--   fitted_model - best model found
--   log_file_text - AutoML debug_log contents
--   workspace - name of the Azure ML workspace where run history is stored
--
-- An example call for a classification problem is:
--    insert into dbo.aml_model(RunId, ExperimentName, Model, LogFileText, WorkspaceName)
--    exec dbo.AutoMLTrain @input_query='
--    SELECT top 100000 
--          CAST([pickup_datetime] AS NVARCHAR(30)) AS pickup_datetime
--          ,CAST([dropoff_datetime] AS NVARCHAR(30)) AS dropoff_datetime
--          ,[passenger_count]
--          ,[trip_time_in_secs]
--          ,[trip_distance]
--          ,[payment_type]
--          ,[tip_class]
--      FROM [dbo].[nyctaxi_sample] order by [hack_license] ',
--      @label_column = 'tip_class',
--      @iterations=10
-- 
-- An example call for forecasting is:
--      insert into dbo.aml_model(RunId, ExperimentName, Model, LogFileText, WorkspaceName)
--      exec dbo.AutoMLTrain @input_query='
--      select cast(timeStamp as nvarchar(30)) as timeStamp,
--             demand,
--      	   precip,
--      	   temp,
--             case when timeStamp < ''2017-01-01'' then 0 else 1 end as is_validate_column
--      from nyc_energy
--      where demand is not null and precip is not null and temp is not null
--      and timeStamp < ''2017-02-01''',
--      @label_column='demand',
--      @task='forecasting',
--      @iterations=10,
--      @iteration_timeout_minutes=5,
--      @time_column_name='timeStamp',
--      @is_validate_column='is_validate_column',
--      @experiment_name='automl-sql-forecast',
--      @primary_metric='normalized_root_mean_squared_error'

SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE OR ALTER PROCEDURE [dbo].[AutoMLTrain]
 (
    @input_query NVARCHAR(MAX),                      -- The SQL Query that will return the data to train and validate the model.
    @label_column NVARCHAR(255)='Label',             -- The name of the column in the result of @input_query that is the label.
    @primary_metric NVARCHAR(40)='AUC_weighted',     -- The metric to optimize.
    @iterations INT=100,                             -- The maximum number of pipelines to train.
    @task NVARCHAR(40)='classification',             -- The type of task.  Can be classification, regression or forecasting.
    @experiment_name NVARCHAR(32)='automl-sql-test', -- This can be used to find the experiment in the Azure Portal.
    @iteration_timeout_minutes INT = 15,             -- The maximum time in minutes for training a single pipeline. 
    @experiment_timeout_minutes INT = 60,            -- The maximum time in minutes for training all pipelines.
    @n_cross_validations INT = 3,                    -- The number of cross validations.
    @blacklist_models NVARCHAR(MAX) = '',            -- A comma separated list of algos that will not be used.
                                                     -- The list of possible models can be found at:
                                                     -- https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train#configure-your-experiment-settings
    @whitelist_models NVARCHAR(MAX) = '',            -- A comma separated list of algos that can be used.
                                                     -- The list of possible models can be found at:
                                                     -- https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train#configure-your-experiment-settings
    @experiment_exit_score FLOAT = 0,                -- Stop the experiment if this score is acheived.
    @sample_weight_column NVARCHAR(255)='',          -- The name of the column in the result of  @input_query that gives a sample weight.
    @is_validate_column NVARCHAR(255)='',            -- The name of the column in the result of  @input_query that indicates if the row is for training or validation.
	                                                 -- In the values of the column, 0 means for training and 1 means for validation.
    @time_column_name  NVARCHAR(255)='',             -- The name of the timestamp column for forecasting.
    @connection_name NVARCHAR(255)='default',        -- The AML connection to use.
    @max_horizon INT = 0                             -- A forecast horizon is a time span into the future (or just beyond the latest date in the training data)
                                                     -- where forecasts of the target quantity are needed.
                                                     -- For example, if data is recorded daily and max_horizon is 5, we will predict 5 days ahead.
 ) AS
BEGIN

    DECLARE @tenantid NVARCHAR(255)
    DECLARE @appid NVARCHAR(255)
    DECLARE @password NVARCHAR(255)
    DECLARE @config_file NVARCHAR(255)

	SELECT @tenantid=TenantId, @appid=AppId, @password=Password, @config_file=ConfigFile
	FROM aml_connection
	WHERE ConnectionName = @connection_name;

	EXEC sp_execute_external_script @language = N'Python', @script = N'import pandas as pd
import logging 
import azureml.core 
import pandas as pd
import numpy as np
from azureml.core.experiment import Experiment 
from azureml.train.automl import AutoMLConfig 
from sklearn import datasets 
import pickle
import codecs
from azureml.core.authentication import ServicePrincipalAuthentication 
from azureml.core.workspace import Workspace 

if __name__.startswith("sqlindb"):
    auth = ServicePrincipalAuthentication(tenantid, appid, password) 
 
    ws = Workspace.from_config(path=config_file, auth=auth) 
 
    project_folder = "./sample_projects/" + experiment_name
 
    experiment = Experiment(ws, experiment_name) 

    data_train = input_data
    X_valid = None
    y_valid = None
    sample_weight_valid = None

    if is_validate_column != "" and is_validate_column is not None:
        data_train = input_data[input_data[is_validate_column] <= 0]
        data_valid = input_data[input_data[is_validate_column] > 0]
        data_train.pop(is_validate_column)
        data_valid.pop(is_validate_column)
        y_valid = data_valid.pop(label_column).values
        if sample_weight_column != "" and sample_weight_column is not None:
            sample_weight_valid = data_valid.pop(sample_weight_column).values
        X_valid = data_valid
        n_cross_validations = None

    y_train = data_train.pop(label_column).values

    sample_weight = None
    if sample_weight_column != "" and sample_weight_column is not None:
        sample_weight = data_train.pop(sample_weight_column).values

    X_train = data_train

    if experiment_timeout_minutes == 0:
        experiment_timeout_minutes = None

    if experiment_exit_score == 0:
        experiment_exit_score = None

    if blacklist_models == "":
        blacklist_models = None

    if blacklist_models is not None:
        blacklist_models = blacklist_models.replace(" ", "").split(",")

    if whitelist_models == "":
        whitelist_models = None

    if whitelist_models is not None:
        whitelist_models = whitelist_models.replace(" ", "").split(",")

    automl_settings = {}
    preprocess = True
    if time_column_name != "" and time_column_name is not None:
        automl_settings = { "time_column_name": time_column_name }
        preprocess = False
        if max_horizon > 0:
            automl_settings["max_horizon"] = max_horizon

    log_file_name = "automl_sqlindb_errors.log"
	 
    automl_config = AutoMLConfig(task = task, 
                                 debug_log = log_file_name, 
                                 primary_metric = primary_metric, 
                                 iteration_timeout_minutes = iteration_timeout_minutes, 
                                 experiment_timeout_minutes = experiment_timeout_minutes,
                                 iterations = iterations, 
                                 n_cross_validations = n_cross_validations, 
                                 preprocess = preprocess,
                                 verbosity = logging.INFO, 
                                 X = X_train,  
                                 y = y_train, 
                                 path = project_folder,
                                 blacklist_models = blacklist_models,
                                 whitelist_models = whitelist_models,
                                 experiment_exit_score = experiment_exit_score,
                                 sample_weight = sample_weight,
                                 X_valid = X_valid,
                                 y_valid = y_valid,
                                 sample_weight_valid = sample_weight_valid,
                                 **automl_settings) 
 
    local_run = experiment.submit(automl_config, show_output = True) 

    best_run, fitted_model = local_run.get_output()

    pickled_model = codecs.encode(pickle.dumps(fitted_model), "base64").decode()

    log_file_text = ""

    try:
        with open(log_file_name, "r") as log_file:
            log_file_text = log_file.read()
    except:
        log_file_text = "Log file not found"

    returned_model = pd.DataFrame({"best_run": [best_run.id], "experiment_name": [experiment_name], "fitted_model": [pickled_model], "log_file_text": [log_file_text], "workspace": [ws.name]}, dtype=np.dtype(np.str))
'
	, @input_data_1 = @input_query
	, @input_data_1_name = N'input_data'
	, @output_data_1_name = N'returned_model'
	, @params = N'@label_column NVARCHAR(255), 
	              @primary_metric NVARCHAR(40),
				  @iterations INT, @task NVARCHAR(40),
				  @experiment_name NVARCHAR(32),
				  @iteration_timeout_minutes INT,
				  @experiment_timeout_minutes INT,
				  @n_cross_validations INT,
				  @blacklist_models NVARCHAR(MAX),
				  @whitelist_models NVARCHAR(MAX),
				  @experiment_exit_score FLOAT,
				  @sample_weight_column NVARCHAR(255),
				  @is_validate_column NVARCHAR(255),
				  @time_column_name  NVARCHAR(255),
				  @tenantid NVARCHAR(255),
				  @appid NVARCHAR(255),
				  @password NVARCHAR(255),
				  @config_file NVARCHAR(255),
				  @max_horizon INT'
	, @label_column = @label_column
	, @primary_metric = @primary_metric
	, @iterations = @iterations
	, @task = @task
	, @experiment_name = @experiment_name
	, @iteration_timeout_minutes = @iteration_timeout_minutes
	, @experiment_timeout_minutes = @experiment_timeout_minutes
	, @n_cross_validations = @n_cross_validations
	, @blacklist_models = @blacklist_models
	, @whitelist_models = @whitelist_models
	, @experiment_exit_score = @experiment_exit_score
	, @sample_weight_column = @sample_weight_column
	, @is_validate_column = @is_validate_column
	, @time_column_name = @time_column_name
	, @tenantid = @tenantid
	, @appid = @appid
	, @password = @password
	, @config_file = @config_file
	, @max_horizon = @max_horizon
WITH RESULT SETS ((best_run NVARCHAR(250), experiment_name NVARCHAR(100), fitted_model VARCHAR(MAX), log_file_text NVARCHAR(MAX), workspace NVARCHAR(100)))
END