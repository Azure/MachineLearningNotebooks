
# Azure Machine Learning - Reinforcement Learning (Public Preview)

<!-- 
Guidelines on README format: https://review.docs.microsoft.com/help/onboard/admin/samples/concepts/readme-template?branch=master

Guidance on onboarding samples to docs.microsoft.com/samples: https://review.docs.microsoft.com/help/onboard/admin/samples/process/onboarding?branch=master

Taxonomies for products and languages: https://review.docs.microsoft.com/new-hope/information-architecture/metadata/taxonomies?branch=master
-->

This is an introduction to the [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/service/) Reinforcement Learning (Public Preview) using the [Ray](https://github.com/ray-project/ray/) framework.

## What is reinforcement learning?

Reinforcement learning is an approach to machine learning to train agents to make a sequence of decisions.  This technique has gained popularity over the last few years as breakthroughs have been made to teach reinforcement learning agents to excel at complex tasks like playing video games.  There are many practical real-world use cases as well, including robotics, chemistry, online recommendations, advertising and more.

In reinforcement learning, the goal is to train an agent *policy* that outputs actions based on the agent’s observations of its environment.  Actions result in further observations and *rewards* for taking the actions.  In reinforcement learning, the full reward for policy actions may take many steps to obtain.  Learning a policy involves many trial-and-error runs of the agent interacting with the environment and improving its policy. 

## Reinforcement learning on Azure Machine Learning

Reinforcement learning support in Azure Machine Learning service enables data scientists to scale training to many powerful CPU or GPU enabled VMs using [Azure Machine Learning compute clusters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#amlcompute) which automatically provision, manage, and scale down these VMs to help manage your costs.

Using these samples, you will learn how to do the following.

1. Use an Azure Machine Learning workspace, set up virtual network and create compute clusters for distributed training.
2. Train reinforcement learning agents using Ray RLlib.

## Contents

| File/folder       | Description                                |
|-------------------|--------------------------------------------|
| [cartpole_ci.ipynb](cartpole-on-compute-instance/cartpole_ci.ipynb)  | Notebook to train a Cartpole playing agent on an Azure Machine Learning Compute Instance |
| [cartpole_sc.ipynb](cartpole-on-single-compute/cartpole_sc.ipynb)  | Notebook to train a Cartpole playing agent on an Azure Machine Learning Compute Cluster (single node) |
| [pong_rllib.ipynb](atari-on-distributed-compute/pong_rllib.ipynb)   | Notebook for distributed training of Pong agent using RLlib on multiple compute targets |
| [minecraft.ipynb](minecraft-on-distributed-compute/minecraft.ipynb)   | Notebook to train an agent to navigate through a lava maze in the Minecraft game |
| [particle.ipynb](multiagent-particle-envs/particle.ipynb)  | Notebook to train policies in a multiagent cooperative navigation scenario based on OpenAI's Particle environments |

## Prerequisites

To make use of these samples, you need the following.

* A Microsoft Azure subscription.
* A Microsoft Azure resource group.
* An Azure Machine Learning Workspace in the resource group.
* Azure Machine Learning training compute. These samples use the VM sizes `STANDARD_NC6` and `STANDARD_D2_V2`.  If these are not available in your region,
you can replace them with other sizes.
* A virtual network set up in the resource group for samples that use multiple compute targets.  The Cartpole and Multi-agent Particle examples do not need a virtual network. Any network security group defined on the virtual network must allow network traffic on ports used by Azure infrastructure services. Sample instructions are provided in Atari Pong and Minecraft example notebooks.


## Setup

You can run these samples in the following ways.

* On an Azure Machine Learning Compute Instance or Azure Data Science Virtual Machine (DSVM).
* On a workstation with Python and the Azure ML Python SDK installed.

### Compute Instance or DSVM
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