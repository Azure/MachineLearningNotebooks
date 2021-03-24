#!/bin/bash

CONDA_ENV_NAME=$1
AUTOML_ENV_FILE=$2
OPTIONS=$3
PIP_NO_WARN_SCRIPT_LOCATION=0
CHECK_CONDA_VERSION_SCRIPT="check_conda_version.py"

if [ "$CONDA_ENV_NAME" == "" ]
then
  CONDA_ENV_NAME="azure_automl"
fi

if [ "$AUTOML_ENV_FILE" == "" ]
then
  AUTOML_ENV_FILE="automl_env_mac.yml"
fi

if [ ! -f $AUTOML_ENV_FILE ]; then
    echo "File $AUTOML_ENV_FILE not found"
    exit 1
fi

if [ ! -f $CHECK_CONDA_VERSION_SCRIPT ]; then
    echo "File $CHECK_CONDA_VERSION_SCRIPT not found"
    exit 1
fi

python "$CHECK_CONDA_VERSION_SCRIPT"
if [ $? -ne 0 ]; then
    exit 1
fi

sed -i '' 's/AZUREML-SDK-VERSION/latest/' $AUTOML_ENV_FILE
brew install libomp

if source activate $CONDA_ENV_NAME 2> /dev/null
then
   echo "Upgrading existing conda environment" $CONDA_ENV_NAME
   pip uninstall azureml-train-automl -y -q
   conda env update --name $CONDA_ENV_NAME --file $AUTOML_ENV_FILE &&
   jupyter nbextension uninstall --user --py azureml.widgets
else
   conda env create -f $AUTOML_ENV_FILE -n $CONDA_ENV_NAME &&
   source activate $CONDA_ENV_NAME &&
   conda install lightgbm -c conda-forge -y &&
   python -m ipykernel install --user --name $CONDA_ENV_NAME --display-name "Python ($CONDA_ENV_NAME)" &&
   jupyter nbextension uninstall --user --py azureml.widgets &&
   echo "" &&
   echo "" &&
   echo "***************************************" &&
   echo "* AutoML setup completed successfully *" &&
   echo "***************************************" &&
   if [ "$OPTIONS" != "nolaunch" ]
   then
      echo "" &&
      echo "Starting jupyter notebook - please run the configuration notebook" &&
      echo "" &&
      jupyter notebook --log-level=50 --notebook-dir '../..'
   fi
fi

if [ $? -gt 0 ]
then
   echo "Installation failed"
fi



