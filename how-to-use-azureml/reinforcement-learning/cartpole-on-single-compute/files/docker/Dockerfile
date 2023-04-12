FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python-opengl \
    rsync \
    xvfb && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/share/man/*

RUN conda install -y conda=23.1.0 python=3.8 && conda clean -ay

RUN conda install -y -c conda-forge ffmpeg=4.0.2
# RUN conda install -c anaconda opencv

RUN pip install ray-on-aml==0.2.1 & \
    pip install --no-cache-dir \
    azureml-defaults \
    azureml-dataset-runtime[fuse,pandas] \
    azureml-contrib-reinforcementlearning \
    gputil \
    scipy \
    pyglet==1.5.27 \
    cloudpickle==1.3.0 \
    tensorboard==2.7.0 \
    tensorflow==2.7.0 \
    tabulate \
    dm_tree \
    lz4 \
    psutil \
    setproctitle \
    pygame \
    gym[classic_control]==0.19.0

RUN pip install protobuf==3.20.0

RUN pip install --upgrade ray==0.8.7 \
    ray[rllib,dashboard,tune]==0.8.7

RUN pip install 'msrest<0.7.0'
