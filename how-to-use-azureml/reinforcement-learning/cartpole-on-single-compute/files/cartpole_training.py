from ray_on_aml.core import Ray_On_AML
import yaml
from ray.tune.tune import run_experiments
from utils import callbacks
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to yaml configuration file')
    args = parser.parse_args()

    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray:  # in the headnode
        ray.init(address="auto")
        print("Configuring run from file: ", args.config)
        experiment_config = None
        with open(args.config, "r") as file:
            experiment_config = yaml.safe_load(file)

        # Set local_dir in each experiment configuration to ensure generated logs get picked up
        # Also set monitor to ensure videos are captured
        for experiment_name, experiment in experiment_config.items():
            experiment["storage_path"] = "./logs"
            experiment['config']['monitor'] = True
        print(f'Config: {experiment_config}')

        trials = run_experiments(
            experiment_config,
            callbacks=[callbacks.TrialCallback()],
            verbose=2
        )
else:
    print("in worker node")
