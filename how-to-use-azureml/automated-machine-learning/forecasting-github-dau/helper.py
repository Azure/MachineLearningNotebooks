import pandas as pd
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.estimator import Estimator
from azureml.core.run import Run
from azureml.automl.core.shared import constants


def split_fraction_by_grain(df, fraction, time_column_name, grain_column_names=None):
    if not grain_column_names:
        df["tmp_grain_column"] = "grain"
        grain_column_names = ["tmp_grain_column"]

    """Group df by grain and split on last n rows for each group."""
    df_grouped = df.sort_values(time_column_name).groupby(
        grain_column_names, group_keys=False
    )

    df_head = df_grouped.apply(
        lambda dfg: dfg.iloc[: -int(len(dfg) * fraction)] if fraction > 0 else dfg
    )

    df_tail = df_grouped.apply(
        lambda dfg: dfg.iloc[-int(len(dfg) * fraction) :] if fraction > 0 else dfg[:0]
    )

    if "tmp_grain_column" in grain_column_names:
        for df2 in (df, df_head, df_tail):
            df2.drop("tmp_grain_column", axis=1, inplace=True)

        grain_column_names.remove("tmp_grain_column")

    return df_head, df_tail


def split_full_for_forecasting(
    df, time_column_name, grain_column_names=None, test_split=0.2
):
    index_name = df.index.name

    # Assumes that there isn't already a column called tmpindex

    df["tmpindex"] = df.index

    train_df, test_df = split_fraction_by_grain(
        df, test_split, time_column_name, grain_column_names
    )

    train_df = train_df.set_index("tmpindex")
    train_df.index.name = index_name

    test_df = test_df.set_index("tmpindex")
    test_df.index.name = index_name

    df.drop("tmpindex", axis=1, inplace=True)

    return train_df, test_df


def get_result_df(remote_run):
    children = list(remote_run.get_children(recursive=True))
    summary_df = pd.DataFrame(
        index=["run_id", "run_algorithm", "primary_metric", "Score"]
    )
    goal_minimize = False
    for run in children:
        if (
            run.get_status().lower() == constants.RunState.COMPLETE_RUN
            and "run_algorithm" in run.properties
            and "score" in run.properties
        ):
            # We only count in the completed child runs.
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


def run_inference(
    test_experiment,
    compute_target,
    script_folder,
    train_run,
    test_dataset,
    lookback_dataset,
    max_horizon,
    target_column_name,
    time_column_name,
    freq,
):
    model_base_name = "model.pkl"
    if "model_data_location" in train_run.properties:
        model_location = train_run.properties["model_data_location"]
        _, model_base_name = model_location.rsplit("/", 1)
    train_run.download_file(
        "outputs/{}".format(model_base_name), "inference/{}".format(model_base_name)
    )
    train_run.download_file("outputs/conda_env_v_1_0_0.yml", "inference/condafile.yml")

    inference_env = Environment("myenv")
    inference_env.docker.enabled = True
    inference_env.python.conda_dependencies = CondaDependencies(
        conda_dependencies_file_path="inference/condafile.yml"
    )

    est = Estimator(
        source_directory=script_folder,
        entry_script="infer.py",
        script_params={
            "--max_horizon": max_horizon,
            "--target_column_name": target_column_name,
            "--time_column_name": time_column_name,
            "--frequency": freq,
            "--model_path": model_base_name,
        },
        inputs=[
            test_dataset.as_named_input("test_data"),
            lookback_dataset.as_named_input("lookback_data"),
        ],
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


def run_multiple_inferences(
    summary_df,
    train_experiment,
    test_experiment,
    compute_target,
    script_folder,
    test_dataset,
    lookback_dataset,
    max_horizon,
    target_column_name,
    time_column_name,
    freq,
):
    for run_name, run_summary in summary_df.iterrows():
        print(run_name)
        print(run_summary)
        run_id = run_summary.run_id
        train_run = Run(train_experiment, run_id)

        test_run = run_inference(
            test_experiment,
            compute_target,
            script_folder,
            train_run,
            test_dataset,
            lookback_dataset,
            max_horizon,
            target_column_name,
            time_column_name,
            freq,
        )

        print(test_run)
        summary_df.loc[summary_df.run_id == run_id, "test_run_id"] = test_run.id

    return summary_df
