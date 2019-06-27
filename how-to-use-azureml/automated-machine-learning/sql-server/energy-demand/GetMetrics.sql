-- This lists all the metrics for all iterations for the most recent run.

DECLARE @RunId NVARCHAR(43)
DECLARE @ExperimentName NVARCHAR(255)

SELECT TOP 1 @ExperimentName=ExperimentName, @RunId=SUBSTRING(RunId, 1, 43)
FROM aml_model
ORDER BY CreatedDate DESC

EXEC dbo.AutoMLGetMetrics @RunId, @ExperimentName