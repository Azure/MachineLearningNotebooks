FROM continuumio/miniconda:4.5.11

# install git
RUN apt-get update && apt-get upgrade -y && apt-get install -y git

# create a new conda environment named azureml
RUN conda create -n azureml -y -q Python=3.6

# install additional packages used by sample notebooks. this is optional
RUN ["/bin/bash", "-c", "source activate azureml && conda install -y tqdm cython matplotlib scikit-learn"]

# install azurmel-sdk components
RUN ["/bin/bash", "-c", "source activate azureml && pip install azureml-sdk[notebooks]==1.0.10"]

# clone Azure ML GitHub sample notebooks
RUN cd /home && git clone -b "azureml-sdk-1.0.10" --single-branch https://github.com/Azure/MachineLearningNotebooks.git

# generate jupyter configuration file
RUN ["/bin/bash", "-c", "source activate azureml && mkdir ~/.jupyter && cd ~/.jupyter && jupyter notebook --generate-config"]

# set an emtpy token for Jupyter to remove authentication. 
# this is NOT recommended for production environment
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

# open up port 8887 on the container
EXPOSE 8887

# start Jupyter notebook server on port 8887 when the container starts
CMD /bin/bash -c "cd /home/MachineLearningNotebooks && source activate azureml && jupyter notebook --port 8887 --no-browser --ip 0.0.0.0 --allow-root"