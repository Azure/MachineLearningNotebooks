import os

import ray
import ray.tune as tune

from utils import callbacks
from minecraft_environment import create_env


def stop(trial_id, result):
    max_train_time = int(os.environ.get("AML_MAX_TRAIN_TIME_SECONDS", 5 * 60 * 60))

    return result["episode_reward_mean"] >= 1 \
        or result["time_total_s"] >= max_train_time


if __name__ == '__main__':
    tune.register_env("Minecraft", create_env)

    ray.init(address='auto')

    tune.run(
        run_or_experiment="IMPALA",
        config={
            "env": "Minecraft",
            "env_config": {
                "mission": "minecraft_missions/lava_maze-v0.xml"
            },
            "num_workers": 10,
            "num_cpus_per_worker": 2,
            "rollout_fragment_length": 50,
            "train_batch_size": 1024,
            "replay_buffer_num_slots": 4000,
            "replay_proportion": 10,
            "learner_queue_timeout": 900,
            "num_sgd_iter": 2,
            "num_data_loader_buffers": 2,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 500000
            },
            "callbacks": {"on_train_result": callbacks.on_train_result},
        },
        stop=stop,
        checkpoint_at_end=True,
        local_dir='./logs'
    )
