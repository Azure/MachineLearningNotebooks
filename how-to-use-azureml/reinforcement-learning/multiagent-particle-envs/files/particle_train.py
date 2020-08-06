import argparse
import re
import os

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_trainable, register_env, get_trainable_cls
import ray.rllib.contrib.maddpg.maddpg as maddpg

from rllib_multiagent_particle_env import env_creator
from util import parse_args


def setup_ray():
    ray.init(address='auto')

    register_env('particle', env_creator)


def gen_policy(args, env, id):
    use_local_critic = [
        args.adv_policy == 'ddpg' if id < args.num_adversaries else
        args.good_policy == 'ddpg' for id in range(env.num_agents)
    ]
    return (
        None,
        env.observation_space_dict[id],
        env.action_space_dict[id],
        {
            'agent_id': id,
            'use_local_critic': use_local_critic[id],
            'obs_space_dict': env.observation_space_dict,
            'act_space_dict': env.action_space_dict,
        }
    )


def gen_policies(args, env_config):
    env = env_creator(env_config)
    return {'policy_%d' % i: gen_policy(args, env, i) for i in range(len(env.observation_space_dict))}


def to_multiagent_config(policies):
    policy_ids = list(policies.keys())
    return {
        'policies': policies,
        'policy_mapping_fn': lambda index: policy_ids[index]
    }


def train(args, env_config):
    def stop(trial_id, result):
        max_train_time = int(os.environ.get('AML_MAX_TRAIN_TIME_SECONDS', 2 * 60 * 60))

        return result['episode_reward_mean'] >= args.final_reward \
            or result['time_total_s'] >= max_train_time

    run_experiments({
        'MADDPG_RLLib': {
            'run': 'contrib/MADDPG',
            'env': 'particle',
            'stop': stop,
            # Uncomment to enable more frequent checkpoints:
            # 'checkpoint_freq': args.checkpoint_freq,
            'checkpoint_at_end': True,
            'local_dir': args.local_dir,
            'restore': args.restore,
            'config': {
                # === Log ===
                'log_level': 'ERROR',

                # === Environment ===
                'env_config': env_config,
                'num_envs_per_worker': args.num_envs_per_worker,
                'horizon': args.max_episode_len,

                # === Policy Config ===
                # --- Model ---
                'good_policy': args.good_policy,
                'adv_policy': args.adv_policy,
                'actor_hiddens': [args.num_units] * 2,
                'actor_hidden_activation': 'relu',
                'critic_hiddens': [args.num_units] * 2,
                'critic_hidden_activation': 'relu',
                'n_step': args.n_step,
                'gamma': args.gamma,

                # --- Exploration ---
                'tau': 0.01,

                # --- Replay buffer ---
                'buffer_size': int(1e6),

                # --- Optimization ---
                'actor_lr': args.lr,
                'critic_lr': args.lr,
                'learning_starts': args.train_batch_size * args.max_episode_len,
                'sample_batch_size': args.sample_batch_size,
                'train_batch_size': args.train_batch_size,
                'batch_mode': 'truncate_episodes',

                # --- Parallelism ---
                'num_workers': args.num_workers,
                'num_gpus': args.num_gpus,
                'num_gpus_per_worker': 0,

                # === Multi-agent setting ===
                'multiagent': to_multiagent_config(gen_policies(args, env_config)),
            },
        },
    }, verbose=1)


if __name__ == '__main__':
    args = parse_args()
    setup_ray()

    env_config = {
        'scenario_name': args.scenario,
        'horizon': args.max_episode_len,
        'video_frequency': args.checkpoint_freq,
    }

    train(args, env_config)
