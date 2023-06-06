import os
import sys
import argparse

from ray.rllib.evaluate import RolloutSaver, rollout
from ray_on_aml.core import Ray_On_AML
import ray.cloudpickle as cloudpickle
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR

from azureml.core import Run
from utils import callbacks

import collections
import copy
import gymnasium as gym
import json
from pathlib import Path


def run_rollout(checkpoint, algo, render, steps, episodes):
    config_dir = os.path.dirname(checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    config = None

    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # Load the config from pickled.
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)
    # If no pkl file found, require command line `--config`.
    else:
        raise ValueError("Could not find params.pkl in either the checkpoint dir or its parent directory")

    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config` (first try from command line, then from
    # pkl file).
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    config = merge_dicts(config, evaluation_config)
    env = config.get("env")

    # Make sure we have evaluation workers.
    if not config.get("evaluation_num_workers"):
        config["evaluation_num_workers"] = config.get("num_workers", 0)
    if not config.get("evaluation_duration"):
        config["evaluation_duration"] = 1

    # Hard-override this as it raises a warning by Algorithm otherwise.
    # Makes no sense anyways, to have it set to None as we don't call
    # `Algorithm.train()` here.
    config["evaluation_interval"] = 1

    # Rendering settings.
    config["render_env"] = render

    # Create the Algorithm from config.
    cls = get_trainable_cls(algo)
    algorithm = cls(env=env, config=config)

    # Load state from checkpoint, if provided.
    if checkpoint:
        algorithm.restore(checkpoint)

    # Do the actual rollout.
    with RolloutSaver(
        outfile=None,
        use_shelve=False,
        write_update_file=False,
        target_steps=steps,
        target_episodes=episodes,
        save_info=False,
    ) as saver:
        rollout(algorithm, env, steps, episodes, saver, not render)
    algorithm.stop()


if __name__ == "__main__":
    # Start ray head (single node)
    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_path', required=True, help='Path to artifacts dataset')
        parser.add_argument('--checkpoint', required=True, help='Name of checkpoint file directory')
        parser.add_argument('--algo', required=True, help='Name of RL algorithm')
        parser.add_argument('--render', default=False, required=False, help='True to render')
        parser.add_argument('--steps', required=False, type=int, help='Number of steps to run')
        parser.add_argument('--episodes', required=False, type=int, help='Number of episodes to run')
        args = parser.parse_args()

        # Get a handle to run
        run = Run.get_context()

        # Get handles to the tarining artifacts dataset and mount path
        dataset_path = run.input_datasets['dataset_path']

        # Find checkpoint file to be evaluated
        checkpoint = os.path.join(dataset_path, args.checkpoint)
        print('Checkpoint:', checkpoint)

        # Start rollout
        ray.init(address='auto')
        run_rollout(checkpoint, args.algo, args.render, args.steps, args.episodes)
