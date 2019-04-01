# Using Databricks as a Compute Target from Azure Machine Learning Pipeline
To use Databricks as a compute target from Azure Machine Learning Pipeline, a DatabricksStep is used. This notebook demonstrates the use of DatabricksStep in Azure Machine Learning Pipeline.

The notebook will show:

1. Running an arbitrary Databricks notebook that the customer has in Databricks workspace
2. Running an arbitrary Python script that the customer has in DBFS
3. Running an arbitrary Python script that is available on local computer (will upload to DBFS, and then run in Databricks)
4. Running a JAR job that the customer has in DBFS.

## Before you begin:
1. **Create an Azure Databricks workspace** in the same subscription where you have your Azure Machine Learning workspace. 
You will need details of this workspace later on to define DatabricksStep. [More information](https://ms.portal.azure.com/#blade/HubsExtension/Resources/resourceType/Microsoft.Databricks%2Fworkspaces).
2. **Create PAT (access token)** at the Azure Databricks portal. [More information](https://docs.databricks.com/api/latest/authentication.html#generate-a-token).
3. **Add demo notebook to ADB** This notebook has a sample you can use as is. Launch Azure Databricks attached to your Azure Machine Learning workspace and add a new notebook.
4. **Create/attach a Blob storage** for use from ADB