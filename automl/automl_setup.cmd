@echo off
set conda_env_name=%1

IF "%conda_env_name%"=="" SET conda_env_name="azure_automl"

call conda activate %conda_env_name% 2>nul:

if not errorlevel 1 (
  call conda env update --file automl_env.yml -n %conda_env_name%
  if errorlevel 1 goto ErrorExit
) else (
  call conda env create -f automl_env.yml -n %conda_env_name%
)

call conda activate %conda_env_name% 2>nul:
if errorlevel 1 goto ErrorExit

call pip install psutil

call jupyter nbextension install --py azureml.train.widgets
if errorlevel 1 goto ErrorExit

call jupyter nbextension enable --py azureml.train.widgets
if errorlevel 1 goto ErrorExit

echo.
echo.
echo ***************************************
echo * AutoML setup completed successfully *
echo ***************************************
echo.
echo Starting jupyter notebook - please run notebook 00.configuration
echo.
jupyter notebook --log-level=50

goto End

:ErrorExit
echo Install failed

:End