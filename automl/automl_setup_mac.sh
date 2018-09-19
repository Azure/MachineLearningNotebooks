#!/bin/bash

CONDA_ENV_NAME=$1

if [ "$CONDA_ENV_NAME" == "" ]
then
  CONDA_ENV_NAME="azure_automl"
fi

if source activate $CONDA_ENV_NAME 2> /dev/null
then
   conda env update -file automl_env.yml -n $CONDA_ENV_NAME
else
   conda env create -f automl_env.yml -n $CONDA_ENV_NAME &&
   source activate $CONDA_ENV_NAME &&
   conda install lightgbm -c conda-forge -y &&
   jupyter nbextension install --py azureml.train.widgets --user &&
   jupyter nbextension enable --py azureml.train.widgets --user &&
   echo "" &&
   echo "" &&
   echo "***************************************" &&
   echo "* AutoML setup completed successfully *" &&
   echo "***************************************" &&
   echo "" &&
   echo "Starting jupyter notebook - please run notebook 00.configuration" &&
   echo "" &&
   jupyter notebook --log-level=50
fi

if [ $? -gt 0 ]
then
   echo "Installation failed"
fi


