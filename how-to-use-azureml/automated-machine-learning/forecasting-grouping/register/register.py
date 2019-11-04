import argparse

from azureml.core import Run, Model

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--model_path")

args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace
print('retrieved ws: {}'.format(ws))

print('begin register model')
model = Model.register(
    workspace=ws,
    model_path=args.model_path,
    model_name=args.model_name
)
print('model registered: {}'.format(model))
print('complete')
