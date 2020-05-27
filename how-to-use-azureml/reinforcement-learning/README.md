
# Azure Machine Learning - Reinforcement Learning (Public Preview)

<!-- 
Guidelines on README format: https://review.docs.microsoft.com/help/onboard/admin/samples/concepts/readme-template?branch=master

Guidance on onboarding samples to docs.microsoft.com/samples: https://review.docs.microsoft.com/help/onboard/admin/samples/process/onboarding?branch=master

Taxonomies for products and languages: https://review.docs.microsoft.com/new-hope/information-architecture/metadata/taxonomies?branch=master
-->

This is an introduction to the [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/service/) Reinforcement Learning (Public Preview) using the [Ray](https://github.com/ray-project/ray/) framework.

Using these samples, you will be able to do the following.

1. Use an Azure Machine Learning workspace, set up virtual network and create compute clusters for running Ray.
2. Run some experiments to train a reinforcement learning agent using Ray and RLlib.

## Contents

| File/folder       | Description                                |
|-------------------|--------------------------------------------|
| [devenv_setup.ipynb](setup/devenv_setup.ipynb) | Notebook to setup development environment for Azure ML RL |
| [cartpole_ci.ipynb](cartpole-on-compute-instance/cartpole_ci.ipynb)  | Notebook to train a Cartpole playing agent on an Azure ML Compute Instance |
| [cartpole_sc.ipynb](cartpole-on-single-compute/cartpole_sc.ipynb)  | Notebook to train a Cartpole playing agent on an Azure ML Compute Cluster (single node) |
| [pong_rllib.ipynb](atari-on-distributed-compute/pong_rllib.ipynb)   | Notebook to train Pong agent using RLlib on multiple compute targets |
| [minecraft.ipynb](minecraft-on-distributed-compute/minecraft.ipynb)   | Notebook to train an agent to navigate through a lava maze in the Minecraft game |

## Prerequisites

To make use of these samples, you need the following.

* A Microsoft Azure subscription.
* A Microsoft Azure resource group.
* An Azure Machine Learning Workspace in the resource group. Please make sure that the VM sizes `STANDARD_NC6` and `STANDARD_D2_V2` are supported in the workspace's region.
* A virtual network set up in the resource group.
  * A virtual network is needed for the examples training on multiple compute targets.
  * The [devenv_setup.ipynb](setup/devenv_setup.ipynb) notebook shows you how to create a virtual network. You can alternatively use an existing virtual network, make sure it's in the same region as workspace is.
  * Any network security group defined on the virtual network must allow network traffic on ports used by Azure infrastructure services. This is described in more detail in the [devenv_setup.ipynb](setup/devenv_setup.ipynb) notebook.


## Setup

You can run these samples in the following ways.

* On an Azure ML Compute Instance or Notebook VM.
* On a workstation with Python and the Azure ML Python SDK installed.

### Azure ML Compute Instance or Notebook VM
#### Update packages


We recommend that you update the required Python packages before you proceed. The following commands are for entering in a Python interpreter such as a notebook.

```shell
# We recommend updating pip to the latest version.
!pip install --upgrade pip
# Update matplotlib for plotting charts
!pip install --upgrade matplotlib
# Update Azure Machine Learning SDK to the latest version
!pip install --upgrade azureml-sdk
# For Jupyter notebook widget used in samples
!pip install --upgrade azureml-widgets
# For Tensorboard used in samples
!pip install --upgrade azureml-tensorboard
# Install Azure Machine Learning Reinforcement Learning SDK
!pip install --upgrade azureml-contrib-reinforcementlearning
```

### Your own workstation
#### Install/update packages

For a local workstation, create a Python environment and install [Azure Machine Learning SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py) and the RL SDK. We recommend Python 3.6 and higher.

```shell
# Activate your environment first.
# e.g.,
# conda activate amlrl
# We recommend updating pip to the latest version.
pip install --upgrade pip
# Install/upgrade matplotlib for plotting charts
pip install --upgrade matplotlib
# Install/upgrade tensorboard used in samples
pip install --upgrade tensorboard
# Install/upgrade Azure ML SDK to the latest version
pip install --upgrade azureml-sdk
# For Jupyter notebook widget used in samples
pip install --upgrade azureml-widgets
# For Tensorboard used in samples
pip install --upgrade azureml-tensorboard
# Install Azure Machine Learning Reinforcement Learning SDK
pip install --upgrade azureml-contrib-reinforcementlearning
# To use the notebook widget, you may need to register and enable the Azure ML extensions first.
jupyter nbextension install --py --user azureml.widgets
jupyter nbextension enable --py --user azureml.widgets
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

For more on SDK concepts, please refer to [notebooks](https://github.com/Azure/MachineLearningNotebooks).

**Please let us know your [feedback](https://github.com/Azure/MachineLearningNotebooks/labels/Reinforcement%20Learning).**

 

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/reinforcement-learning/README.png)