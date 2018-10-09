# Training ML models with Azure ML SDK
These notebook tutorials cover the various scenarios for training machine learning and deep learning models with Azure Machine Learning.

## Sample notebooks
- [01.train-hyperparameter-tune-deploy-with-pytorch](./01.train-hyperparameter-tune-deploy-with-pytorch/01.train-hyperparameter-tune-deploy-with-pytorch.ipynb)  
Train, hyperparameter tune, and deploy a PyTorch image classification model that distinguishes bees vs. ants using transfer learning. Azure ML concepts covered:
  - Create a remote compute target (Batch AI cluster)
  - Upload training data using `Datastore`
  - Run a single-node `PyTorch` training job
  - Hyperparameter tune model with HyperDrive
  - Find and register the best model
  - Deploy model to ACI
- [02.distributed-pytorch-with-horovod](./02.distributed-pytorch-with-horovod/02.distributed-pytorch-with-horovod.ipynb)  
Train a PyTorch model on the MNIST dataset using distributed training with Horovod. Azure ML concepts covered:
  - Create a remote compute target (Batch AI cluster)
  - Run a two-node distributed `PyTorch` training job using Horovod
- [03.train-hyperparameter-tun-deploy-with-tensorflow](./03.train-hyperparameter-tune-deploy-with-tensorflow/03.train-hyperparameter-tune-deploy-with-tensorflow.ipynb)  
Train, hyperparameter tune, and deploy a TensorFlow model on the MNIST dataset. Azure ML concepts covered:
  - Create a remote compute target (Batch AI cluster)
  - Upload training data using `Datastore`
  - Run a single-node `TensorFlow` training job
  - Leverage features of the `Run` object
  - Download the trained model
  - Hyperparameter tune model with HyperDrive
  - Find and register the best model
  - Deploy model to ACI
- [04.distributed-tensorflow-with-horovod](./04.distributed-tensorflow-with-horovod/04.distributed-tensorflow-with-horovod.ipynb)  
Train a TensorFlow word2vec model using distributed training with Horovod. Azure ML concepts covered:
  - Create a remote compute target (Batch AI cluster)
  - Upload training data using `Datastore`
  - Run a two-node distributed `TensorFlow` training job using Horovod
- [05.distributed-tensorflow-with-parameter-server](./05.distributed-tensorflow-with-parameter-server/05.distributed-tensorflow-with-parameter-server.ipynb)  
Train a TensorFlow model on the MNIST dataset using native distributed TensorFlow (parameter server). Azure ML concepts covered:
  - Create a remote compute target (Batch AI cluster)
  - Run a two workers, one parameter server distributed `TensorFlow` training job
- [06.distributed-cntk-with-custom-docker](./06.distributed-cntk-with-custom-docker/06.distributed-cntk-with-custom-docker.ipynb)  
Train a CNTK model on the MNIST dataset using the Azure ML base `Estimator` with custom Docker image and distributed training. Azure ML concepts covered:
  - Create a remote compute target (Batch AI cluster)
  - Upload training data using `Datastore`
  - Run a base `Estimator` training job using a custom Docker image from Docker Hub
  - Distributed CNTK two-node training job via MPI using base `Estimator`
  
- [07.tensorboard](./07.tensorboard/07.tensorboard.ipynb)  
Train a TensorFlow MNIST model locally, on a DSVM, and on Batch AI and view the logs live on TensorBoard. Azure ML concepts covered:
  - Run the training job locally with Azure ML and run TensorBoard locally. Start (and stop) an Azure ML `TensorBoard` object to stream and view the logs
  - Run the training job on a remote DSVM and stream the logs to TensorBoard
  - Run the training job on a remote Batch AI cluster and stream the logs to TensorBoard
  - Start a `Tensorboard` instance that displays the logs from all three above runs in one
- [08.export-run-history-to-tensorboard](./08.export-run-history-to-tensorboard/08.export-run-history-to-tensorboard.ipynb)
  - Start an Azure ML `Experiment` and log metrics to `Run` history
  - Export the `Run` history logs to TensorBoard logs
  - View the logs in TensorBoard
