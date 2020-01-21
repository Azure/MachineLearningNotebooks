import argparse
import json

from azureml.core import Run, Model
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice


script_file_name = 'score.py'
conda_env_file_name = 'myenv.yml'

print("In deploy.py")
parser = argparse.ArgumentParser()
parser.add_argument("--time_column_name", type=str, help="time column name")
parser.add_argument("--group_column_names", type=str, help="group column names")
parser.add_argument("--model_names", type=str, help="model names")
parser.add_argument("--service_name", type=str, help="service name")

args = parser.parse_args()

# replace the group column names in scoring script to the ones set by user
print("Update group_column_names")
print(args.group_column_names)

with open(script_file_name, 'r') as cefr:
    content = cefr.read()
with open(script_file_name, 'w') as cefw:
    content = content.replace('<<groups>>', args.group_column_names.rstrip())
    cefw.write(content.replace('<<time_colname>>', args.time_column_name.rstrip()))

with open(script_file_name, 'r') as cefr1:
    content1 = cefr1.read()
print(content1)

model_list = json.loads(args.model_names)
print(model_list)

run = Run.get_context()
ws = run.experiment.workspace

myenv = Environment.from_conda_specification(name="env", file_path=conda_env_file_name)

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=2,
    tags={"method": "grouping"},
    description='grouping demo aci deployment'
)

inference_config = InferenceConfig(entry_script=script_file_name, environment=myenv)

models = []
for model_name in model_list:
    models.append(Model(ws, name=model_name))

service = Model.deploy(
    ws,
    name=args.service_name,
    models=models,
    inference_config=inference_config,
    deployment_config=deployment_config
)
service.wait_for_deployment(True)
