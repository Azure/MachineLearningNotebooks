import argparse
import os
import re

from rllib_multiagent_particle_env import CUSTOM_SCENARIOS


def parse_args():
    parser = argparse.ArgumentParser('MADDPG with OpenAI MPE')

    # Environment
    parser.add_argument('--scenario', type=str, default='simple',
                        choices=['simple', 'simple_speaker_listener',
                                 'simple_crypto', 'simple_push',
                                 'simple_tag', 'simple_spread', 'simple_adversary'
                                 ] + CUSTOM_SCENARIOS,
                        help='name of the scenario script')
    parser.add_argument('--max-episode-len', type=int, default=25,
                        help='maximum episode length')
    parser.add_argument('--num-episodes', type=int, default=60000,
                        help='number of episodes')
    parser.add_argument('--num-adversaries', type=int, default=0,
                        help='number of adversaries')
    parser.add_argument('--good-policy', type=str, default='maddpg',
                        help='policy for good agents')
    parser.add_argument('--adv-policy', type=str, default='maddpg',
                        help='policy of adversaries')

    # Core training parameters
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='discount factor')
    # NOTE: 1 iteration = sample_batch_size * num_workers timesteps * num_envs_per_worker
    parser.add_argument('--sample-batch-size', type=int, default=25,
                        help='number of data points sampled /update /worker')
    parser.add_argument('--train-batch-size', type=int, default=1024,
                        help='number of data points /update')
    parser.add_argument('--n-step', type=int, default=1,
                        help='length of multistep value backup')
    parser.add_argument('--num-units', type=int, default=64,
                        help='number of units in the mlp')
    parser.add_argument('--final-reward', type=int, default=-400,
                        help='final reward after which to stop training')

    # Checkpoint
    parser.add_argument('--checkpoint-freq', type=int, default=200,
                        help='save model once every time this many iterations are completed')
    parser.add_argument('--local-dir', type=str, default='./logs',
                        help='path to save checkpoints')
    parser.add_argument('--restore', type=str, default=None,
                        help='directory in which training state and model are loaded')

    # Parallelism
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-envs-per-worker', type=int, default=4)
    parser.add_argument('--num-gpus', type=int, default=0)

    return parser.parse_args()


def find_final_checkpoint(start_dir):
    def find(pattern, path):
        result = []
        for root, _, files in os.walk(path):
            for name in files:
                if pattern.match(name):
                    result.append(os.path.join(root, name))
        return result

    cp_pattern = re.compile('.*checkpoint-\\d+$')
    checkpoint_files = find(cp_pattern, start_dir)

    checkpoint_numbers = []
    for file in checkpoint_files:
        checkpoint_numbers.append(int(file.split('-')[-1]))

    final_checkpoint_number = max(checkpoint_numbers)

    return next(
        checkpoint_file for checkpoint_file in checkpoint_files
        if checkpoint_file.endswith(str(final_checkpoint_number)))
