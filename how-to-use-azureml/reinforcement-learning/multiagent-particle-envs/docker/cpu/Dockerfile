FROM akdmsft/particle-cpu

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

RUN pip3 install --upgrade \
    ray==0.8.7 \
    ray[rllib]==0.8.7 \
    ray[tune]==0.8.7