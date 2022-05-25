import argparse
import os
import uuid
import shutil
from azureml.core.model import Model, Dataset
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace
import azureml.automl.core.shared.constants as constants
from azureml.train.automl.run import AutoMLRun


def get_best_automl_run(pipeline_run):
    all_children = [c for c in pipeline_run.get_children()]
    automl_step = [
        c for c in all_children if c.properties.get("runTemplate") == "AutoML"
    ]
    for c in all_children:
        print(c, c.properties)
    automlrun = AutoMLRun(pipeline_run.experiment, automl_step[0].id)
    best = automlrun.get_best_child()
    return best


def get_model_path(model_artifact_path):
    return model_artifact_path.split("/")[1]


if __name__ == "__main__":
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
    new_dir = str(uuid.uuid4())
    os.makedirs(new_dir)

    # Register model with training dataset
    best_run = get_best_automl_run(run.parent)
    model_artifact_path = best_run.properties[constants.PROPERTY_KEY_OF_MODEL_PATH]
    algo = best_run.properties.get("run_algorithm")
    model_artifact_dir = model_artifact_path.split("/")[0]
    model_file_name = model_artifact_path.split("/")[1]
    model = best_run.register_model(
        args.model_name,
        model_path=model_artifact_dir,
        datasets=datasets,
        tags={"algorithm": algo, "model_file_name": model_file_name},
    )

    print("Registered version {0} of model {1}".format(model.version, model.name))
