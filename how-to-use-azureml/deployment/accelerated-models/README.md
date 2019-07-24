
# Notebooks for Microsoft Azure Machine Learning Hardware Accelerated Models SDK

Easily create and train a model using various deep neural networks (DNNs) as a featurizer for deployment to Azure or a Data Box Edge device for ultra-low latency inferencing using FPGA's. These models are currently available:

* ResNet 50
* ResNet 152
* DenseNet-121
* VGG-16
* SSD-VGG  

To learn more about the azureml-accel-model classes, see the section [Model Classes](#model-classes) below or the [Azure ML Accel Models SDK documentation](https://docs.microsoft.com/en-us/python/api/azureml-accel-models/azureml.accel?view=azure-ml-py).

### Step 1: Create an Azure ML workspace
Follow [these instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace) to install the Azure ML SDK on your local machine, create an Azure ML workspace, and set up your notebook environment, which is required for the next step.

### Step 2: Check your FPGA quota
Use the Azure CLI to check whether you have quota.

```shell
az vm list-usage --location "eastus" -o table
```

The other locations are ``southeastasia``, ``westeurope``, and ``westus2``.

Under the "Name" column, look for "Standard PBS Family vCPUs" and ensure you have at least 6 vCPUs under "CurrentValue."

If you do not have quota, then submit a request form [here](https://aka.ms/accelerateAI).

### Step 3: Install the Azure ML Accelerated Models SDK
Once you have set up your environment, install the Azure ML Accel Models SDK. This package requires tensorflow >= 1.6,<2.0 to be installed. 

If you already have tensorflow >= 1.6,<2.0 installed in your development environment, you can install the SDK package using: 

```
pip install azureml-accel-models
```

If you do not have tensorflow >= 1.6,<2.0 and are using a CPU-only development environment, our SDK with tensorflow can be installed using:

```
pip install azureml-accel-models[cpu]
```

If your machine supports GPU (for example, on an [Azure DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/overview)), then you can leverage the tensorflow-gpu functionality using:

```
pip install azureml-accel-models[gpu]
```

### Step 4: Follow our notebooks

The notebooks in this repo walk through the following scenarios: 
* [Quickstart](accelerated-models-quickstart.ipynb), deploy and inference a ResNet50 model trained on ImageNet
* [Object Detection](accelerated-models-object-detection.ipynb), deploy and inference an SSD-VGG model that can do object detection
* [Training models](accelerated-models-training.ipynb), train one of our accelerated models on the Kaggle Cats and Dogs dataset to see how to improve accuracy on custom datasets

<a name="model-classes"></a>
## Model Classes
As stated above, we support 5 Accelerated Models. Here's more information on their input and output tensors.

**Available models and output tensors**

The available models and the corresponding default classifier output tensors are below. This is the value that you would use during inferencing if you used the default classifier.
* Resnet50, QuantizedResnet50 
``
output_tensors = "classifier_1/resnet_v1_50/predictions/Softmax:0"
``
* Resnet152, QuantizedResnet152 
``
output_tensors = "classifier/resnet_v1_152/predictions/Softmax:0"
``
* Densenet121, QuantizedDensenet121
``
output_tensors = "classifier/densenet121/predictions/Softmax:0"
``
* Vgg16, QuantizedVgg16 
``
output_tensors = "classifier/vgg_16/fc8/squeezed:0"
``
* SsdVgg, QuantizedSsdVgg
``
output_tensors = ['ssd_300_vgg/block4_box/Reshape_1:0', 'ssd_300_vgg/block7_box/Reshape_1:0', 'ssd_300_vgg/block8_box/Reshape_1:0', 'ssd_300_vgg/block9_box/Reshape_1:0', 'ssd_300_vgg/block10_box/Reshape_1:0', 'ssd_300_vgg/block11_box/Reshape_1:0', 'ssd_300_vgg/block4_box/Reshape:0', 'ssd_300_vgg/block7_box/Reshape:0', 'ssd_300_vgg/block8_box/Reshape:0', 'ssd_300_vgg/block9_box/Reshape:0', 'ssd_300_vgg/block10_box/Reshape:0', 'ssd_300_vgg/block11_box/Reshape:0']
``

For more information, please reference the azureml.accel.models package in the [Azure ML Python SDK documentation](https://docs.microsoft.com/en-us/python/api/azureml-accel-models/azureml.accel.models?view=azure-ml-py).

**Input tensors**

The input_tensors value defaults to "Placeholder:0" and is created in the [Image Preprocessing](#construct-model) step in the line: 
``
in_images = tf.placeholder(tf.string)
``

You can change the input_tensors name by doing this: 
``
in_images = tf.placeholder(tf.string, name="images")
``


## Resources
*  [Read more about FPGAs](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-accelerate-with-fpgas)