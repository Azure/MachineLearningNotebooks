-- This shows using the AutoMLTrain stored procedure to create a forecasting model for the nyc_energy dataset.

INSERT INTO dbo.aml_model(RunId, ExperimentName, Model, LogFileText, WorkspaceName)
EXEC dbo.AutoMLTrain @input_query='
SELECT CAST(timeStamp as NVARCHAR(30)) as timeStamp,
       demand,
	   precip,
	   temp,
	   CASE WHEN timeStamp < ''2017-01-01'' THEN 0 ELSE 1 END AS is_validate_column
FROM nyc_energy
WHERE demand IS NOT NULL AND precip IS NOT NULL AND temp IS NOT NULL
and timeStamp < ''2017-02-01''',
@label_column='demand',
@task='forecasting',
@iterations=10,
@iteration_timeout_minutes=5,
@time_column_name='timeStamp',
@is_validate_column='is_validate_column',
@experiment_name='automl-sql-forecast',
@primary_metric='normalized_root_mean_squared_error'

