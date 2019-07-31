-- This shows using the AutoMLForecast stored procedure to predict using a forecasting model for the nyc_energy dataset.

DECLARE @Model NVARCHAR(MAX) = (SELECT TOP 1 Model FROM dbo.aml_model
                                WHERE ExperimentName = 'automl-sql-forecast'
								ORDER BY CreatedDate DESC)

DECLARE @max_horizon INT = 48
DECLARE @split_time NVARCHAR(22) = (SELECT DATEADD(hour, -@max_horizon, MAX(timeStamp)) FROM nyc_energy WHERE demand IS NOT NULL)

DECLARE @TestDataQuery NVARCHAR(MAX) = '
SELECT CAST(timeStamp AS NVARCHAR(30)) AS timeStamp,
       demand,
	   precip,
	   temp
FROM nyc_energy
WHERE demand IS NOT NULL AND precip IS NOT NULL AND temp IS NOT NULL
AND timeStamp > ''' + @split_time + ''''

EXEC dbo.AutoMLForecast @input_query=@TestDataQuery,
@label_column='demand',
@time_column_name='timeStamp',
@model=@model
WITH RESULT SETS ((timeStamp DATETIME, grain NVARCHAR(255), predicted_demand FLOAT, precip FLOAT, temp FLOAT, actual_demand FLOAT))
