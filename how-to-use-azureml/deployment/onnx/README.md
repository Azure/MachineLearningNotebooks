# ONNX on Azure Machine Learning

These tutorials show how to create and deploy Open Neural Network eXchange ([ONNX](http://onnx.ai)) models in Azure Machine Learning environments using [ONNX Runtime](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx) for inference. Once deployed as a web service, you can ping the model with your own set of images to be analyzed!

## Tutorials

0. [Configure your Azure Machine Learning Workspace](../../../configuration.ipynb)

#### Obtain models from the [ONNX Model Zoo](https://github.com/onnx/models) and deploy with ONNX Runtime Inference
1. [Handwritten Digit Classification (MNIST)](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-mnist-deploy.ipynb)
2. [Facial Expression Recognition (Emotion FER+)](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-facial-expression-recognition-deploy.ipynb)

#### Demo Notebooks from Microsoft Ignite 2018
Note that the following notebooks do not have evaluation sections for the models since they were deployed as part of a live demo. You can find the respective pre-processing and post-processing code linked from the ONNX Model Zoo Github pages ([ResNet](https://github.com/onnx/models/tree/master/models/image_classification/resnet), [TinyYoloV2](https://github.com/onnx/models/tree/master/tiny_yolov2)), or experiment with the ONNX models by [running them in the browser](https://microsoft.github.io/onnxjs-demo/#/).

3. [Image Recognition (ResNet50)](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-modelzoo-aml-deploy-resnet50.ipynb)
4. [Convert Core ML Model to ONNX and deploy - Real Time Object Detection (TinyYOLO)](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-convert-aml-deploy-tinyyolo.ipynb) 

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
