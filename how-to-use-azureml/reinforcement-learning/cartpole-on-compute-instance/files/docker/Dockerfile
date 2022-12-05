FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

USER root
RUN conda install -c anaconda python=3.7

RUN pip install ray-on-aml==0.1.6
RUN pip install gym[atari]==0.19.0
RUN pip install gym[accept-rom-license]==0.19.0
RUN pip install ale-py==0.7.0
RUN pip install azureml-core
RUN pip install azureml-dataset-runtime
RUN pip install ray==0.8.7
RUN pip install ray[rllib,tune,serve]==0.8.7
RUN pip install tensorflow==1.14.0
RUN pip install 'msrest<0.7.0'

RUN apt-get update
RUN apt-get install -y jq
RUN apt-get install -y rsync
