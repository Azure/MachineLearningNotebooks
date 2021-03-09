Azure Synapse Analytics is a limitless analytics service that brings together data integration, enterprise data warehousing, and big data analytics. It gives you the freedom to query data on your terms, using either serverless or dedicated resources—at scale. Azure Synapse brings these worlds together with a unified experience to ingest, explore, prepare, manage, and serve data for immediate BI and machine learning needs. A core offering within Azure Synapse Analytics are serverless Apache Spark pools enhanced for big data workloads.  

Synapse in Aml integration is for customers who want to use Apache Spark in Azure Synapse Analytics to prepare data at scale in Azure ML before training their ML model. This will allow customers to work on their end-to-end ML lifecycle including large-scale data preparation, model training and deployment within Azure ML workspace without having to use suboptimal tools for machine learning or switch between multiple tools for data preparation and model training. The ability to perform all ML tasks within Azure ML will reduce time required for customers to iterate on a machine learning project which typically includes multiple rounds of data preparation and training.

In the public preview, the capabilities are provided:

- Link Azure Synapse Analytics workspace to Azure Machine Learning workspace (via ARM, UI or SDK) 
- Attach Apache Spark pools powered by Azure Synapse Analytics as Azure Machine Learning compute targets (via ARM, UI or SDK) 
- Launch Apache Spark sessions in notebooks and perform interactive data exploration and preparation.  This interactive experience leverages Apache Spark magic and customers will have session-level Conda support to install packages. 
- Productionize ML pipelines by leveraging Apache Spark pools to pre-process big data 

# Using Synapse in Azure machine learning

## Create synapse resources

Follow up the documents to create Synapse workspace and resource-setup.sh is available for you to create the resources.

- Create from [Portal](https://docs.microsoft.com/en-us/azure/synapse-analytics/quickstart-create-workspace)
- Create from [Cli](https://docs.microsoft.com/en-us/azure/synapse-analytics/quickstart-create-workspace-cli)

Follow up the documents to create Synapse spark pool

- Create from [Portal](https://docs.microsoft.com/en-us/azure/synapse-analytics/quickstart-create-apache-spark-pool-portal)
- Create from [Cli](https://docs.microsoft.com/en-us/cli/azure/ext/synapse/synapse/spark/pool?view=azure-cli-latest)

## Link Synapse Workspace

Make sure you are the owner of synapse workspace so that you can link synapse workspace into AML.
You can run resource-setup.py to link the synapse workspace and attach compute

```python
from azureml.core import Workspace
ws = Workspace.from_config()

from azureml.core import LinkedService, SynapseWorkspaceLinkedServiceConfiguration
synapse_link_config = SynapseWorkspaceLinkedServiceConfiguration(
    subscription_id="<subscription id>",
    resource_group="<resource group",
    name="<synapse workspace name>"
)

linked_service = LinkedService.register(
    workspace=ws,
    name='<link name>',
    linked_service_config=synapse_link_config)

```

## Attach synapse spark pool as AzureML compute

```python

from azureml.core.compute import SynapseCompute, ComputeTarget
spark_pool_name = "<spark pool name>"
attached_synapse_name = "<attached compute name>"

attach_config = SynapseCompute.attach_configuration(
        linked_service,
        type="SynapseSpark",
        pool_name=spark_pool_name)

synapse_compute=ComputeTarget.attach(
        workspace=ws,
        name=attached_synapse_name,
        attach_configuration=attach_config)

synapse_compute.wait_for_completion()
```

## Set up permission

Grant Spark admin role to system assigned identity of the linked service so that the user can submit experiment run or pipeline run from AML workspace to synapse spark pool.

Grant Spark admin role to the specific user so that the user can start spark session to synapse spark pool.

You can get the system assigned identity information by running

```python
print(linked_service.system_assigned_identity_principal_id)
```

- Launch synapse studio of the synapse workspace and grant linked service MSI "Synapse Apache Spark administrator" role.

- In azure portal grant linked service MSI "Storage Blob Data Contributor" role of the primary adlsgen2 account of synapse workspace to use the library management feature.
