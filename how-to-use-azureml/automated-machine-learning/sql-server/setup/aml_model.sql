-- This is a table to hold the results from the AutoMLTrain procedure.
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[aml_model](
    [Id] [int] IDENTITY(1,1) NOT NULL PRIMARY KEY,
    [Model] [varchar](max) NOT NULL,        -- The model, which can be passed to AutoMLPredict for testing or prediction.
    [RunId] [nvarchar](250) NULL,           -- The RunId, which can be used to view the model in the Azure Portal.
    [CreatedDate] [datetime] NULL,
    [ExperimentName] [nvarchar](100) NULL,  -- Azure ML Experiment Name
    [WorkspaceName] [nvarchar](100) NULL,   -- Azure ML Workspace Name
	[LogFileText] [nvarchar](max) NULL
) 
GO

ALTER TABLE [dbo].[aml_model] ADD  DEFAULT (getutcdate()) FOR [CreatedDate]
GO


