import pandas as pd
from azureml.core import Environment
from azureml.train.estimator import Estimator
from azureml.core.run import Run


def run_inference(
    test_experiment,
    compute_target,
    script_folder,
    train_run,
    test_dataset,
    target_column_name,
    model_name,
):

    inference_env = train_run.get_environment()

    est = Estimator(
        source_directory=script_folder,
        entry_script="infer.py",
        script_params={
            "--target_column_name": target_column_name,
            "--model_name": model_name,
        },
        inputs=[test_dataset.as_named_input("test_data")],
        compute_target=compute_target,
        environment_definition=inference_env,
    )

    run = test_experiment.submit(
        est,
        tags={
            "training_run_id": train_run.id,
            "run_algorithm": train_run.properties["run_algorithm"],
            "valid_score": train_run.properties["score"],
            "primary_metric": train_run.properties["primary_metric"],
        },
    )

    run.log("run_algorithm", run.tags["run_algorithm"])
    return run


def get_result_df(remote_run):

    children = list(remote_run.get_children(recursive=True))
    summary_df = pd.DataFrame(
        index=["run_id", "run_algorithm", "primary_metric", "Score"]
    )
    goal_minimize = False
    for run in children:
        if "run_algorithm" in run.properties and "score" in run.properties:
            summary_df[run.id] = [
                run.id,
                run.properties["run_algorithm"],
                run.properties["primary_metric"],
                float(run.properties["score"]),
            ]
            if "goal" in run.properties:
                goal_minimize = run.properties["goal"].split("_")[-1] == "min"

    summary_df = summary_df.T.sort_values(
        "Score", ascending=goal_minimize
    ).drop_duplicates(["run_algorithm"])
    summary_df = summary_df.set_index("run_algorithm")

    return summary_df
