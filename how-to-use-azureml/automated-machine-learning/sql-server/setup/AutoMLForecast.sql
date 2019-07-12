-- This procedure forecast values based on a forecasting model returned by AutoMLTrain.
-- It returns a dataset with the forecasted values.
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE OR ALTER PROCEDURE [dbo].[AutoMLForecast]
 (
   @input_query NVARCHAR(MAX),          -- A SQL query returning data to predict on.
   @model NVARCHAR(MAX),                -- A model returned from AutoMLTrain.
   @time_column_name  NVARCHAR(255)='', -- The name of the timestamp column for forecasting.
   @label_column  NVARCHAR(255)='',     -- Optional name of the column from input_query, which should be ignored when predicting
   @y_query_column NVARCHAR(255)='',    -- Optional value column that can be used for predicting.
                                        -- If specified, this can contain values for past times (after the model was trained)
									    -- and contain Nan for future times.
   @forecast_column_name NVARCHAR(255) = 'predicted'
                                        -- The name of the output column containing the forecast value.
 ) AS 
BEGIN 
  
    EXEC sp_execute_external_script @language = N'Python', @script = N'import pandas as pd 
import azureml.core  
import numpy as np 
from azureml.train.automl import AutoMLConfig  
import pickle 
import codecs 
  
model_obj = pickle.loads(codecs.decode(model.encode(), "base64")) 
  
test_data = input_data.copy() 

if label_column != "" and label_column is not None:
    y_test = test_data.pop(label_column).values
else:
    y_test = None 

if y_query_column != "" and y_query_column is not None:
    y_query = test_data.pop(y_query_column).values
else:
    y_query = np.repeat(np.nan, len(test_data))

X_test = test_data 

if time_column_name != "" and time_column_name is not None:
    X_test[time_column_name] = pd.to_datetime(X_test[time_column_name])

y_fcst, X_trans = model_obj.forecast(X_test, y_query) 

def align_outputs(y_forecast, X_trans, X_test, y_test, forecast_column_name):
    # Demonstrates how to get the output aligned to the inputs
    # using pandas indexes. Helps understand what happened if
    # the output shape differs from the input shape, or if
    # the data got re-sorted by time and grain during forecasting.
    
    # Typical causes of misalignment are:
    # * we predicted some periods that were missing in actuals -> drop from eval
    # * model was asked to predict past max_horizon -> increase max horizon
    # * data at start of X_test was needed for lags -> provide previous periods

    df_fcst = pd.DataFrame({forecast_column_name : y_forecast})
    # y and X outputs are aligned by forecast() function contract
    df_fcst.index = X_trans.index
    
    # align original X_test to y_test    
    X_test_full = X_test.copy()
    if y_test is not None:
        X_test_full[label_column] = y_test

    # X_test_full does not include origin, so reset for merge
    df_fcst.reset_index(inplace=True)
    X_test_full = X_test_full.reset_index().drop(columns=''index'')
    together = df_fcst.merge(X_test_full, how=''right'')
    
    # drop rows where prediction or actuals are nan 
    # happens because of missing actuals 
    # or at edges of time due to lags/rolling windows
    clean = together[together[[label_column, forecast_column_name]].notnull().all(axis=1)]
    return(clean)

combined_output = align_outputs(y_fcst, X_trans, X_test, y_test, forecast_column_name)
  
' 
    , @input_data_1 = @input_query 
    , @input_data_1_name = N'input_data' 
    , @output_data_1_name = N'combined_output' 
    , @params = N'@model NVARCHAR(MAX), @time_column_name  NVARCHAR(255), @label_column NVARCHAR(255), @y_query_column NVARCHAR(255), @forecast_column_name NVARCHAR(255)' 
    , @model = @model 
	, @time_column_name = @time_column_name
	, @label_column = @label_column
	, @y_query_column = @y_query_column
	, @forecast_column_name = @forecast_column_name
END
