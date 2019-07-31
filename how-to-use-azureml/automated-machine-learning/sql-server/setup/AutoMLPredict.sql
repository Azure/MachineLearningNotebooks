-- This procedure predicts values based on a model returned by AutoMLTrain and a dataset.
-- It returns the dataset with a new column added, which is the predicted value.
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE OR ALTER PROCEDURE [dbo].[AutoMLPredict]
 (
   @input_query NVARCHAR(MAX),      -- A SQL query returning data to predict on.
   @model NVARCHAR(MAX),            -- A model returned from AutoMLTrain.
   @label_column  NVARCHAR(255)=''  -- Optional name of the column from input_query, which should be ignored when predicting
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
X_test = test_data 
  
predicted = model_obj.predict(X_test) 
  
combined_output = input_data.assign(predicted=predicted)
  
' 
    , @input_data_1 = @input_query 
    , @input_data_1_name = N'input_data' 
    , @output_data_1_name = N'combined_output' 
    , @params = N'@model NVARCHAR(MAX), @label_column  NVARCHAR(255)' 
    , @model = @model 
	, @label_column = @label_column
END
