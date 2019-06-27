-- This procedure returns a list of metrics for each iteration of a run.
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE OR ALTER PROCEDURE [dbo].[AutoMLGetMetrics]
 (
	@run_id NVARCHAR(250),                           -- The RunId
    @experiment_name NVARCHAR(32)='automl-sql-test', -- This can be used to find the experiment in the Azure Portal.
    @connection_name NVARCHAR(255)='default'         -- The AML connection to use.
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
import numpy as np
from azureml.core.experiment import Experiment 
from azureml.train.automl.run import AutoMLRun
from azureml.core.authentication import ServicePrincipalAuthentication 
from azureml.core.workspace import Workspace 

auth = ServicePrincipalAuthentication(tenantid, appid, password) 
 
ws = Workspace.from_config(path=config_file, auth=auth) 
 
experiment = Experiment(ws, experiment_name) 

ml_run = AutoMLRun(experiment = experiment, run_id = run_id)

children = list(ml_run.get_children())
iterationlist = []
metricnamelist = []
metricvaluelist = []

for run in children:
    properties = run.get_properties()
    if "iteration" in properties:
        iteration = int(properties["iteration"])
        for metric_name, metric_value in run.get_metrics().items():
            if isinstance(metric_value, float):
                iterationlist.append(iteration)
                metricnamelist.append(metric_name)
                metricvaluelist.append(metric_value)
             
metrics = pd.DataFrame({"iteration": iterationlist, "metric_name": metricnamelist, "metric_value": metricvaluelist})
'
    , @output_data_1_name = N'metrics'
	, @params = N'@run_id NVARCHAR(250), 
				  @experiment_name NVARCHAR(32),
  				  @tenantid NVARCHAR(255),
				  @appid NVARCHAR(255),
				  @password NVARCHAR(255),
				  @config_file NVARCHAR(255)'
    , @run_id = @run_id
	, @experiment_name = @experiment_name
	, @tenantid = @tenantid
	, @appid = @appid
	, @password = @password
	, @config_file = @config_file
WITH RESULT SETS ((iteration INT, metric_name NVARCHAR(100), metric_value FLOAT))
END