If you get an error when you go to submit your run that says that sklearn is not found:
- Cause: It is probably because, when you created your workspace, the yml file was overwrited with a default from azureml.
- Solution: do a "git pull origin master" after your workspace is completed. Make sure that the conda_dependencies.yml file has scikit-learn as a pip package. Resubmit the run.