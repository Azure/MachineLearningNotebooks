-- This is a table to store the Azure ML connection information.
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[aml_connection](
    [Id] [int] IDENTITY(1,1) NOT NULL PRIMARY KEY,
	[ConnectionName] [nvarchar](255) NULL,
	[TenantId] [nvarchar](255) NULL,
	[AppId] [nvarchar](255) NULL,
	[Password] [nvarchar](255) NULL,
	[ConfigFile] [nvarchar](255) NULL
) ON [PRIMARY]
GO


