import argparse
import os
import sys

import ray
from ray.rllib import rollout
from ray.tune.registry import get_trainable_cls

from azureml.core import Run

from utils import callbacks


DEFAULT_RAY_ADDRESS = 'localhost:6379'


def run_rollout(args, parser, ray_address):

    config = args.config
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init(address=ray_address)

    # Create the Trainer from config.
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)

    # Load state from checkpoint.
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    num_episodes = int(args.episodes)

    # Determine the video output directory.
    use_arg_monitor = False
    try:
        args.video_dir
    except AttributeError:
        print("There is no such attribute: args.video_dir")
        use_arg_monitor = True

    video_dir = None
    if not use_arg_monitor:
        if args.monitor:
            video_dir = os.path.join("./logs", "video")
        elif args.video_dir:
            video_dir = os.path.expanduser(args.video_dir)

    # Do the actual rollout.
    with rollout.RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info) as saver:
        if use_arg_monitor:
            rollout.rollout(
                agent,
                args.env,
                num_steps,
                num_episodes,
                saver,
                args.no_render,
                args.monitor)
        else:
            rollout.rollout(
                agent, args.env,
                num_steps,
                num_episodes,
                saver,
                args.no_render, video_dir)


if __name__ == "__main__":

    # Add positional argument - serves as placeholder for checkpoint
    argvc = sys.argv[1:]
    argvc.insert(0, 'checkpoint-placeholder')

    # Parse arguments
    rollout_parser = rollout.create_parser()

    rollout_parser.add_argument(
        '--checkpoint-number', required=False, type=int, default=1,
        help='Checkpoint number of the checkpoint from which to roll out')

    rollout_parser.add_argument(
        '--ray-address', required=False, default=DEFAULT_RAY_ADDRESS,
        help='The address of the Ray cluster to connect to')

    args = rollout_parser.parse_args(argvc)

    # Get a handle to run
    run = Run.get_context()

    # Get handles to the tarining artifacts dataset and mount path
    artifacts_dataset = run.input_datasets['artifacts_dataset']
    artifacts_path = run.input_datasets['artifacts_path']

    # Find checkpoint file to be evaluated
    checkpoint_id = '-' + str(args.checkpoint_number)
    checkpoint_files = list(filter(
        lambda filename: filename.endswith(checkpoint_id),
        artifacts_dataset.to_path()))

    checkpoint_file = checkpoint_files[0]
    if checkpoint_file[0] == '/':
        checkpoint_file = checkpoint_file[1:]
    checkpoint = os.path.join(artifacts_path, checkpoint_file)
    print('Checkpoint:', checkpoint)

    # Set rollout checkpoint
    args.checkpoint = checkpoint

    # Start rollout
    run_rollout(args, rollout_parser, args.ray_address)
