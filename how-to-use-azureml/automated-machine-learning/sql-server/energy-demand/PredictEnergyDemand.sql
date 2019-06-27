-- This shows using the AutoMLPredict stored procedure to predict using a forecasting model for the nyc_energy dataset.

DECLARE @Model NVARCHAR(MAX) = (SELECT TOP 1 Model FROM dbo.aml_model
                                WHERE ExperimentName = 'automl-sql-forecast'
								ORDER BY CreatedDate DESC)

EXEC dbo.AutoMLPredict @input_query='
SELECT CAST(timeStamp AS NVARCHAR(30)) AS timeStamp,
       demand,
	   precip,
	   temp
FROM nyc_energy
WHERE demand IS NOT NULL AND precip IS NOT NULL AND temp IS NOT NULL
AND timeStamp >= ''2017-02-01''',
@label_column='demand',
@model=@model
WITH RESULT SETS ((timeStamp NVARCHAR(30), actual_demand FLOAT, precip FLOAT, temp FLOAT, predicted_demand FLOAT))