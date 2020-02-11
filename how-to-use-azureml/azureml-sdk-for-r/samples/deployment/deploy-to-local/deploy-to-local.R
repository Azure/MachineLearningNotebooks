# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

# Register model and deploy locally
# This example shows how to deploy a web service in step-by-step fashion:
#   
# 1) Register model
# 2) Deploy the model as a web service in a local Docker container.
# 3) Invoke web service with SDK or call web service with raw HTTP call.
# 4) Quickly test changes to your entry script by reloading the local service.
# 5) Optionally, you can also make changes to model and update the local service.

library(azuremlsdk)
library(jsonlite)

ws <- load_workspace_from_config()

# Register the model
model <- register_model(ws, model_path = "project_files/model.rds",
                        model_name = "model.rds")

# Create environment
r_env <- r_environment(name = "r_env")

# Create inference config
inference_config <- inference_config(
  entry_script = "score.R",
  source_directory = "project_files",
  environment = r_env)

# Create local deployment config
local_deployment_config <- local_webservice_deployment_config()

# Deploy the web service
# NOTE:
# The Docker image runs as a Linux container. If you are running Docker for Windows, you need to ensure the Linux Engine is running:
# # PowerShell command to switch to Linux engine
# & 'C:\Program Files\Docker\Docker\DockerCli.exe' -SwitchLinuxEngine
service <- deploy_model(ws,
                        'rservice-local',
                        list(model),
                        inference_config,
                        local_deployment_config)
# Wait for deployment
wait_for_deployment(service, show_output = TRUE)

# Show the port of local service
message(service$port)

# If you encounter any issue in deploying the webservice, please visit
# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-troubleshoot-deployment

# Inferencing
# versicolor
# plant <- data.frame(Sepal.Length = 6.4,
#                     Sepal.Width = 2.8,
#                     Petal.Length = 4.6,
#                     Petal.Width = 1.8)
# setosa
plant <- data.frame(Sepal.Length = 5.1,
                    Sepal.Width = 3.5,
                    Petal.Length = 1.4,
                    Petal.Width = 0.2)
# # virginica
# plant <- data.frame(Sepal.Length = 6.7,
#                     Sepal.Width = 3.3,
#                     Petal.Length = 5.2,
#                     Petal.Width = 2.3)

#Test the web service
invoke_webservice(service, toJSON(plant))

## The last few lines of the logs should have the correct prediction and should display -> R[write to console]: "setosa" 
cat(gsub(pattern = "\n", replacement = " \n", x = get_webservice_logs(service)))

## Test the web service with a HTTP Raw request
# 
# NOTE:
# To test the service locally use the https://localhost:<local_service$port> URL

# Import the request library
library(httr)
# Get the service scoring URL from the service object, its URL is for testing locally
local_service_url <- service$scoring_uri #Same as https://localhost:<local_service$port>

#POST request to web service
resp <- POST(local_service_url, body = plant, encode = "json", verbose())

## The last few lines of the logs should have the correct prediction and should display -> R[write to console]: "setosa" 
cat(gsub(pattern = "\n", replacement = " \n", x = get_webservice_logs(service)))


# Optional, use a new scoring script
inference_config <- inference_config(
  entry_script = "score_new.R",
  source_directory = "project_files",
  environment = r_env)

## Then reload the service to see the changes made
reload_local_webservice_assets(service)

## Check reloaded service, you will see the last line will say "this is a new scoring script! I was reloaded"
invoke_webservice(service, toJSON(plant))
cat(gsub(pattern = "\n", replacement = " \n", x = get_webservice_logs(service)))

# Update service
# If you want to change your model(s), environment, or deployment configuration, call update() to rebuild the Docker image.

# update_local_webservice(service, models = [NewModelObject], deployment_config = deployment_config, wait = FALSE, inference_config = inference_config)

# Delete service
delete_local_webservice(service)
