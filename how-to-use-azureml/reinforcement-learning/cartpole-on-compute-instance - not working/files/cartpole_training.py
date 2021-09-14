import argparse
import os
import sys

import ray
from ray.rllib import train
from ray import tune

from utils import callbacks


DEFAULT_RAY_ADDRESS = 'localhost:6379'


if __name__ == "__main__":

    # Parse arguments and add callbacks to config
    train_parser = train.create_parser()

    args = train_parser.parse_args()
    args.config["callbacks"] = {"on_train_result": callbacks.on_train_result}

    # Trace if video capturing is on
    if 'monitor' in args.config and args.config['monitor']:
        print("Video capturing is ON!")

    # Start (connect to) Ray cluster
    if args.ray_address is None:
        args.ray_address = DEFAULT_RAY_ADDRESS

    ray.init(address=args.ray_address)

    # Run training task using tune.run
    tune.run(
        run_or_experiment=args.run,
        config=dict(args.config, env=args.env),
        stop=args.stop,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=args.checkpoint_at_end,
        local_dir=args.local_dir
    )
