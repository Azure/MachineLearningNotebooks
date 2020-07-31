FROM mcr.microsoft.com/azureml/base:openmpi3.1.2-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
	cpio \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    gcc \
    xvfb \
    python-opengl \
    x11-xserver-utils \
    ffmpeg \
    mesa-utils \
    nano \
    vim \
    rsync \
 && rm -rf /var/lib/apt/lists/*

# Install python 3.7
RUN conda install python==3.7 

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Install required pip packages
RUN pip install --upgrade pip setuptools && pip install --upgrade \
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
    cloudpickle==1.2.0 \
    tabulate \
    dm_tree \
    lz4 \
    opencv-python \
    ray==0.8.3 \
    ray[rllib]==0.8.3 \
    ray[tune]==0.8.3

# Install particle
RUN git clone https://github.com/openai/multiagent-particle-envs.git
COPY patch_files/* multiagent-particle-envs/multiagent/
RUN cd multiagent-particle-envs && \
    pip install -e . && \
    pip install --upgrade pyglet==1.3.2
