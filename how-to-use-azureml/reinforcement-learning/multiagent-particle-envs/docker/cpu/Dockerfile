# DisableDockerDetector "Disabled to unblock PRs until the owner can fix the file. Not used in any prod deployments - only as a documentation for the customers"
FROM akdmsft/particle-cpu

RUN conda install -c anaconda python=3.7

# Install required pip packages
RUN pip3 install --upgrade pip setuptools && pip3 install --upgrade \
    pandas \
    matplotlib \
    psutil \
    numpy \
    scipy \
    gym \
    azureml-defaults \
    tensorboardX \
    tensorflow==1.15 \
    tensorflow-probability==0.8.0 \
    onnxruntime \
    tf2onnx \
    cloudpickle==1.1.1 \
    tabulate \
    dm_tree \
    lz4 \
    opencv-python

RUN cd multiagent-particle-envs && \
    pip3 install -e . && \
    pip3 install --upgrade pyglet==1.3.2

RUN pip3 install ray-on-aml==0.1.6

RUN pip install protobuf==3.20.0

RUN pip3 install --upgrade \
    ray==0.8.7 \
    ray[rllib]==0.8.7 \
    ray[tune]==0.8.7

RUN pip install 'msrest<0.7.0'