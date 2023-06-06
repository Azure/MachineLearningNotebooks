FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python-opengl \
    rsync \
    xvfb && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/share/man/*

RUN pip install ray-on-aml==0.2.4 \
    ray==2.4.0 \
    ray[rllib]==2.4.0 \
    mlflow==2.3.1 \
    azureml-defaults==1.50.0 \
    azureml-dataset-runtime[fuse,pandas]==1.50.0 \
    azureml-contrib-reinforcementlearning==1.50.0 \
    gputil==1.4.0 \
    scipy==1.9.1 \
    pyglet==2.0.6 \
    cloudpickle==2.2.1 \
    tensorflow==2.11.0 \
    tensorflow-probability==0.19.0 \
    torch \
    tabulate==0.9.0 \
    dm_tree==0.1.8 \
    lz4==4.3.2 \
    psutil==5.9.4 \
    setproctitle==1.3.2 \
    pygame==2.1.0 \
    gymnasium[classic_control]==0.26.3 \
    gym[classic_control]==0.26.2

# Display the exact versions we have installed
RUN pip freeze
