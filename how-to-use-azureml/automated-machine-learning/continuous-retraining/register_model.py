from azureml.core.model import Model, Dataset
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--model_path")
parser.add_argument("--ds_name")
args = parser.parse_args()

print("Argument 1(model_name): %s" % args.model_name)
print("Argument 2(model_path): %s" % args.model_path)
print("Argument 3(ds_name): %s" % args.ds_name)

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

train_ds = Dataset.get_by_name(ws, args.ds_name)
datasets = [(Dataset.Scenario.TRAINING, train_ds)]

# Register model with training dataset

model = Model.register(
    workspace=ws,
    model_path=args.model_path,
    model_name=args.model_name,
    datasets=datasets,
)

print("Registered version {0} of model {1}".format(model.version, model.name))
