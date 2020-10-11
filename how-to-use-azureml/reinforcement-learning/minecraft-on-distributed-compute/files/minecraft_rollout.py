import argparse
import os
import re

from azureml.core import Run
from azureml.core.model import Model

from minecraft_environment import create_env_for_rollout
from malmo_video_recorder import MalmoVideoRecorder
from gym import wrappers

import ray
import ray.tune as tune
from ray.rllib import rollout
from ray.tune.registry import get_trainable_cls


def write_mission_file_for_seed(mission_file, seed):
    with open(mission_file, 'r') as base_file:
        mission_file_path = mission_file.replace('v0', seed)
        content = base_file.read().format(SEED_PLACEHOLDER=seed)

        mission_file = open(mission_file_path, 'w')
        mission_file.writelines(content)
        mission_file.close()

    return mission_file_path


def run_rollout(trainable_type, mission_file, seed):
    # Writes the mission file for minerl
    mission_file_path = write_mission_file_for_seed(mission_file, seed)

    # Instantiate the agent.  Note: the IMPALA trainer implementation in
    # Ray uses an AsyncSamplesOptimizer.  Under the hood, this starts a
    # LearnerThread which will wait for training samples.  This will fail
    # after a timeout, but has no influence on the rollout. See
    # https://github.com/ray-project/ray/blob/708dff6d8f7dd6f7919e06c1845f1fea0cca5b89/rllib/optimizers/aso_learner.py#L66
    config = {
        "env_config": {
            "mission": mission_file_path,
            "is_rollout": True,
            "seed": seed
        },
        "num_workers": 0
    }
    cls = get_trainable_cls(args.run)
    agent = cls(env="Minecraft", config=config)

    # The optimizer is not needed during a rollout
    agent.optimizer.stop()

    # Load state from checkpoint
    agent.restore(f'{checkpoint_path}/{checkpoint_file}')

    # Get a reference to the environment
    env = agent.workers.local_worker().env

    # Let the agent choose actions until the game is over
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)

        total_reward += reward

    print(f'Total reward using seed {seed}: {total_reward}')

    # This avoids a sigterm trace in the logs, see minerl.env.malmo.Instance
    env.instance.watcher_process.kill()

    env.close()
    agent.stop()

    return env.get_trajectory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--run', required=False, default="IMPALA")
    args = parser.parse_args()

    # Register custom Minecraft environment
    tune.register_env("Minecraft", create_env_for_rollout)

    ray.init(address='auto')

    # Download the model files (contains a checkpoint)
    ws = Run.get_context().experiment.workspace
    model = Model(ws, args.model_name)
    checkpoint_path = model.download(exist_ok=True)

    files_ = os.listdir(checkpoint_path)
    cp_pattern = re.compile('^checkpoint-\\d+$')

    checkpoint_file = None
    for f_ in files_:
        if cp_pattern.match(f_):
            checkpoint_file = f_

    if checkpoint_file is None:
        raise Exception("No checkpoint file found.")

    # These are the Minecraft mission seeds for the rollouts
    rollout_seeds = ['1234', '43289', '65224', '983341']

    # Initialize the Malmo video recorder
    video_recorder = MalmoVideoRecorder()
    video_recorder.init_malmo()

    # Path references to the mission files
    base_training_mission_file = \
        'minecraft_missions/lava_maze_rollout-v0.xml'
    base_video_recording_mission_file = \
        'minecraft_missions/lava_maze_rollout_video.xml'

    for seed in rollout_seeds:
        trajectory = run_rollout(
            args.run,
            base_training_mission_file,
            seed)

        video_recorder.record_malmo_video(
            trajectory,
            base_video_recording_mission_file,
            seed)
