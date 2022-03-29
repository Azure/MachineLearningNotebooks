from azureml.core import ScriptRunConfig


def run_rolling_forecast(
    test_experiment,
    compute_target,
    train_run,
    test_dataset,
    target_column_name,
    inference_folder="./forecast",
):
    train_run.download_file("outputs/model.pkl", inference_folder + "/model.pkl")

    inference_env = train_run.get_environment()

    config = ScriptRunConfig(
        source_directory=inference_folder,
        script="forecasting_script.py",
        arguments=[
            "--target_column_name",
            target_column_name,
            "--test_dataset",
            test_dataset.as_named_input(test_dataset.name),
        ],
        compute_target=compute_target,
        environment=inference_env,
    )

    run = test_experiment.submit(
        config,
        tags={
            "training_run_id": train_run.id,
            "run_algorithm": train_run.properties["run_algorithm"],
            "valid_score": train_run.properties["score"],
            "primary_metric": train_run.properties["primary_metric"],
        },
    )

    run.log("run_algorithm", run.tags["run_algorithm"])
    return run
