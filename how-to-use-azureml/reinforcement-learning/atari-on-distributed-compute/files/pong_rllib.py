import ray
import ray.tune as tune
from ray.rllib import train

import os
import sys

from azureml.core import Run
from utils import callbacks

DEFAULT_RAY_ADDRESS = 'localhost:6379'

if __name__ == "__main__":

    # Parse arguments
    train_parser = train.create_parser()

    args = train_parser.parse_args()
    print("Algorithm config:", args.config)

    if args.ray_address is None:
        args.ray_address = DEFAULT_RAY_ADDRESS

    ray.init(address=args.ray_address)

    tune.run(
        run_or_experiment=args.run,
        config={
            "env": args.env,
            "num_gpus": args.config["num_gpus"],
            "num_workers": args.config["num_workers"],
            "callbacks": {"on_train_result": callbacks.on_train_result},
            "sample_batch_size": 50,
            "train_batch_size": 1000,
            "num_sgd_iter": 2,
            "num_data_loader_buffers": 2,
            "model": {"dim": 42},
        },
        stop=args.stop,
      #  local_dir='./logs',
       checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=args.checkpoint_at_end,
        local_dir=args.local_dir
    )
