from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.estimator import Estimator
from azureml.core.run import Run


def run_rolling_forecast(test_experiment, compute_target, train_run, test_dataset,
                         max_horizon, target_column_name, time_column_name,
                         freq='D', inference_folder='./forecast'):
    condafile = inference_folder + '/condafile.yml'
    train_run.download_file('outputs/model.pkl',
                            inference_folder + '/model.pkl')
    train_run.download_file('outputs/conda_env_v_1_0_0.yml', condafile)

    inference_env = Environment("myenv")
    inference_env.docker.enabled = True
    inference_env.python.conda_dependencies = CondaDependencies(
        conda_dependencies_file_path=condafile)

    est = Estimator(source_directory=inference_folder,
                    entry_script='forecasting_script.py',
                    script_params={
                        '--max_horizon': max_horizon,
                        '--target_column_name': target_column_name,
                        '--time_column_name': time_column_name,
                        '--frequency': freq
                    },
                    inputs=[test_dataset.as_named_input('test_data')],
                    compute_target=compute_target,
                    environment_definition=inference_env)

    run = test_experiment.submit(est,
                                 tags={
                                     'training_run_id': train_run.id,
                                     'run_algorithm': train_run.properties['run_algorithm'],
                                     'valid_score': train_run.properties['score'],
                                     'primary_metric': train_run.properties['primary_metric']
                                 })

    run.log("run_algorithm", run.tags['run_algorithm'])
    return run
