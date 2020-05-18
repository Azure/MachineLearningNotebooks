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

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Install Minecraft needed libraries
RUN mkdir -p /usr/share/man/man1 && \
    sudo apt-get update && \
    sudo apt-get install -y \
    openjdk-8-jre-headless=8u162-b12-1 \
    openjdk-8-jdk-headless=8u162-b12-1 \
    openjdk-8-jre=8u162-b12-1 \
    openjdk-8-jdk=8u162-b12-1

# Create a Python 3.7 environment
RUN conda install conda-build \
    && conda create -y --name py37 python=3.7.3 \
    && conda clean -ya
ENV CONDA_DEFAULT_ENV=py37

# Install minerl
RUN pip install --upgrade --user minerl

RUN pip install \
    pandas \
    matplotlib \
    numpy \
    scipy \
    azureml-defaults \
    tensorboardX \
    tensorflow==1.15rc2 \
    tabulate \
    dm_tree \
    lz4 \
    ray==0.8.3 \
    ray[rllib]==0.8.3 \
    ray[tune]==0.8.3

COPY patch_files/* /root/.local/lib/python3.7/site-packages/minerl/env/Malmo/Minecraft/src/main/java/com/microsoft/Malmo/Client/

# Start minerl to pre-fetch minerl files (saves time when starting minerl during training)
RUN xvfb-run -a -s "-screen 0 1400x900x24" python -c "import gym; import minerl; env = gym.make('MineRLTreechop-v0'); env.close();"

RUN pip install --index-url https://test.pypi.org/simple/ malmo && \
    python -c "import malmo.minecraftbootstrap; malmo.minecraftbootstrap.download();"

ENV MALMO_XSD_PATH="/app/MalmoPlatform/Schemas"
