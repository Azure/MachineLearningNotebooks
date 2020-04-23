# ONNX on Azure Machine Learning

These tutorials show how to create and deploy Open Neural Network eXchange ([ONNX](http://onnx.ai)) models in Azure Machine Learning environments using [ONNX Runtime](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx) for inference. Once deployed as a web service, you can ping the model with your own set of images to be analyzed!

## Tutorials

0. If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, [Configure your Azure Machine Learning Workspace](../../../configuration.ipynb)

#### Obtain pretrained models from the [ONNX Model Zoo](https://github.com/onnx/models) and deploy with ONNX Runtime
1. [MNIST - Handwritten Digit Classification with ONNX Runtime](onnx-inference-mnist-deploy.ipynb)
2. [Emotion FER+ - Facial Expression Recognition with ONNX Runtime](onnx-inference-facial-expression-recognition-deploy.ipynb)

#### Train model on Azure ML, convert to ONNX, and deploy with ONNX Runtime
3. [MNIST - Train using PyTorch and deploy with ONNX Runtime](onnx-train-pytorch-aml-deploy-mnist.ipynb)

#### Demo Notebooks from Microsoft Ignite 2018
Note that the following notebooks do not have evaluation sections for the models since they were deployed as part of a live demo. You can find the respective pre-processing and post-processing code linked from the ONNX Model Zoo Github pages ([ResNet](https://github.com/onnx/models/tree/master/models/image_classification/resnet), [TinyYoloV2](https://github.com/onnx/models/tree/master/tiny_yolov2)), or experiment with the ONNX models by [running them in the browser](https://microsoft.github.io/onnxjs-demo/#/).

4. [ResNet50 - Image Recognition with ONNX Runtime](onnx-modelzoo-aml-deploy-resnet50.ipynb)
5. [TinyYoloV2 - Convert from CoreML and deploy with ONNX Runtime](onnx-convert-aml-deploy-tinyyolo.ipynb)

## Documentation
- [ONNX Runtime Python API Documentation](http://aka.ms/onnxruntime-python)
- [Azure Machine Learning API Documentation](http://aka.ms/aml-docs)

## Related Articles
- [Building and Deploying ONNX Runtime Models](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx)
- [Azure AI – Making AI Real for Business](https://aka.ms/aml-blog-overview)
- [What’s new in Azure Machine Learning](https://aka.ms/aml-blog-whats-new)

## License
Copyright (c) Microsoft Corporation. All rights reserved.  
Licensed under the MIT License.

## Acknowledgements
These tutorials were developed by Vinitra Swamy and Prasanth Pulavarthi of the Microsoft AI Frameworks team and adapted for presentation at Microsoft Ignite 2018.


 ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/deployment/onnx/README.png)
