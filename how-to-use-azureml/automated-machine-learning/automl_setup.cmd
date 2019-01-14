@echo off
set conda_env_name=%1
set automl_env_file=%2
set PIP_NO_WARN_SCRIPT_LOCATION=0

IF "%conda_env_name%"=="" SET conda_env_name="azure_automl"
IF "%automl_env_file%"=="" SET automl_env_file="automl_env.yml"

IF NOT EXIST %automl_env_file% GOTO YmlMissing

call conda activate %conda_env_name% 2>nul:

if not errorlevel 1 (
  echo Upgrading azureml-sdk[automl,notebooks,explain] in existing conda environment %conda_env_name%
  call pip install --upgrade azureml-sdk[automl,notebooks,explain]
  if errorlevel 1 goto ErrorExit
) else (
  call conda env create -f %automl_env_file% -n %conda_env_name%
)

call conda activate %conda_env_name% 2>nul:
if errorlevel 1 goto ErrorExit

call python -m ipykernel install --user --name %conda_env_name% --display-name "Python (%conda_env_name%)"

REM azureml.widgets is now installed as part of the pip install under the conda env.
REM Removing the old user install so that the notebooks will use the latest widget.
call jupyter nbextension uninstall --user --py azureml.widgets

echo.
echo.
echo ***************************************
echo * AutoML setup completed successfully *
echo ***************************************
echo.
echo Starting jupyter notebook - please run the configuration notebook 
echo.
jupyter notebook --log-level=50 --notebook-dir='..\..'

goto End

:YmlMissing
echo File %automl_env_file% not found.

:ErrorExit
echo Install failed

:End