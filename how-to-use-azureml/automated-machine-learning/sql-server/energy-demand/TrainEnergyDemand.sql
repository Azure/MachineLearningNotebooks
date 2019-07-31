-- This shows using the AutoMLTrain stored procedure to create a forecasting model for the nyc_energy dataset.

DECLARE @max_horizon INT = 48
DECLARE @split_time NVARCHAR(22) = (SELECT DATEADD(hour, -@max_horizon, MAX(timeStamp)) FROM nyc_energy WHERE demand IS NOT NULL)

DECLARE @TrainDataQuery NVARCHAR(MAX) = '
SELECT CAST(timeStamp as NVARCHAR(30)) as timeStamp,
       demand,
	   precip,
	   temp
FROM nyc_energy
WHERE demand IS NOT NULL AND precip IS NOT NULL AND temp IS NOT NULL
and timeStamp < ''' + @split_time + ''''

INSERT INTO dbo.aml_model(RunId, ExperimentName, Model, LogFileText, WorkspaceName)
EXEC dbo.AutoMLTrain @input_query= @TrainDataQuery,
@label_column='demand',
@task='forecasting',
@iterations=10,
@iteration_timeout_minutes=5,
@time_column_name='timeStamp',
@max_horizon=@max_horizon,
@experiment_name='automl-sql-forecast',
@primary_metric='normalized_root_mean_squared_error'

