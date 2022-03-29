from ray_on_aml.core import Ray_On_AML

import ray.tune as tune
from ray.rllib import train

from utils import callbacks

if __name__ == "__main__":

    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray:  # in the headnode
        # Parse arguments
        train_parser = train.create_parser()

        args = train_parser.parse_args()
        print("Algorithm config:", args.config)

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
            local_dir='./logs')
    else:
        print("in worker node")
