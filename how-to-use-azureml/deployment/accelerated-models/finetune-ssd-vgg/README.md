# Fine-tuning a VGG-SSD Object Detector and Deploying it on FPGA in the Cloud

## Overview

The `notebooks` subfolder contains two notebooks that demonstrate fine-tuning and deploying an object detector. 

Fine-tuning is covered in [Finetune VGG SSD](notebooks/Finetune%20VGG%20SSD.ipynb) notebook. It shows the end-to-end process of creating a new object detector based on the [VGG SSD architecture](https://www.cs.unc.edu/~wliu/papers/ssd.pdf) trained on VOC07 + 12 (**REVIEW**: verify) datasets. 

The [Deploy Accelerated]("notebooks/Deploy%20Accelerated.ipynb") notebook shows how to use AzureML for deploying the model obtained by fine-tuning the VGG SSD detector on the FPGA cloud. Refer to this [README](../README.md) for instructions on how to install AzureML and get access to FPGA-enabled machines in Azure.

## Preparation

We recommend the following configuration before getting started with training:

* CUDA 10.0 + cuDNN 7.4
* [Anaconda Python](https://www.anaconda.com/distribution/)
* Create a conda [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). E.g.:

```sh
$ conda create -n myenv python=3.6 anaconda
$ conda activate myenv
```
* Install Tensorflow 1.10+ with GPU support in the environment. E.g.:

```sh
pip install -U --ignore-installed tensorflow-gpu==1.13.1
```
* Install opencv-python

```sh
$ pip install opencv-python
```

* Make sure the steps from this [README](../README.md) are complete.


## Training

[Finetune VGG SSD](notebooks/Finetune%20VGG%20SSD.ipynb) demonstrates all aspects of fine-tuning VGG SSD. The model can be fine-tuned on a dataset with objects categorized into a maximum of 20 classes (21 if counting "none" or "background"), due to current FPGA deployment limitations.

### The Code

All of the modules necessary for the training process are located in the `tfssd` folder. It needs to be added to the system path prior to running training.

```python
sys.path.insert(0, os.path.abspath('tfssd'))
```

### Preparing Training/Validation Data

The detector understands data in [PASCAL VOC](https://gist.github.com/Prasad9/30900b0ef1375cc7385f4d85135fdb44). Tools like [labelImg](https://github.com/tzutalin/labelImg) produce data in this format. See the `examples` subfolder for a sample small dataset. In this example we train our detector to recognize gaps between products on a general store shelves with a goal to be able to alert the store manager quickly that re-stocking is necessary.

![](images/annotated.png)

Each image file needs to have a matching XML file. The files should be placed into `JPEGImages` and `Annotations` directories. `check_labelmatch` in `dataset_utils` module will ensure that there are no "orphaned" images or annotations. See `sample.jpg` and `sample.xml` (shown above).

`pascalvoc_to_tfrecords.run` converts images and their annotations into TFRecord format.

**NOTE**: We expect the dataset to be split into "training" and "validation" parts. This is done in the notebook by calling `train_test_split` from `sklearn.model`