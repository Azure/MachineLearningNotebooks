import json
import logging

import gym
import minerl.env.core
import minerl.env.comms
import numpy as np

from ray.rllib.env.atari_wrappers import FrameStack
from minerl.env.malmo import InstanceManager

# Modify the MineRL timeouts to detect common errors
# quicker and speed up recovery
minerl.env.core.SOCKTIME = 60.0
minerl.env.comms.retry_timeout = 1


class EnvWrapper(minerl.env.core.MineRLEnv):
    def __init__(self, xml, port):
        InstanceManager.configure_malmo_base_port(port)
        self.action_to_command_array = [
            'move 1',
            'camera 0 270',
            'camera 0 90']

        super().__init__(
            xml,
            gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            gym.spaces.Discrete(3)
        )

        self.metadata['video.frames_per_second'] = 2

    def _setup_spaces(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def _process_action(self, action_in) -> str:
        assert self.action_space.contains(action_in)
        assert action_in <= len(
            self.action_to_command_array) - 1, 'action index out of bounds.'
        return self.action_to_command_array[action_in]

    def _process_observation(self, pov, info):
        '''
        Overwritten to simplify: returns only `pov` and
        not as the MineRLEnv an obs_dict (observation directory)
        '''
        pov = np.frombuffer(pov, dtype=np.uint8)

        if pov is None or len(pov) == 0:
            raise Exception('Invalid observation, probably an aborted peek')
        else:
            pov = pov.reshape(
                (self.height, self.width, self.depth)
            )[::-1, :, :]

        assert self.observation_space.contains(pov)

        self._last_pov = pov

        return pov


class TrackingEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._actions = [
            self._forward,
            self._turn_left,
            self._turn_right
        ]

    def _reset_state(self):
        self._facing = (1, 0)
        self._position = (0, 0)
        self._visited = {}
        self._update_visited()

    def _forward(self):
        self._position = (
            self._position[0] + self._facing[0],
            self._position[1] + self._facing[1]
        )

    def _turn_left(self):
        self._facing = (self._facing[1], -self._facing[0])

    def _turn_right(self):
        self._facing = (-self._facing[1], self._facing[0])

    def _encode_state(self):
        return self._position

    def _update_visited(self):
        state = self._encode_state()
        value = self._visited.get(state, 0)
        self._visited[state] = value + 1
        return value

    def reset(self):
        self._reset_state()
        return super().reset()

    def step(self, action):
        o, r, d, i = super().step(action)
        self._actions[action]()
        revisit_count = self._update_visited()
        if revisit_count == 0:
            r += 0.1

        return o, r, d, i


class TrajectoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._trajectory = []
        self._action_to_malmo_command_array = ['move 1', 'turn -1', 'turn 1']

    def get_trajectory(self):
        return self._trajectory

    def _to_malmo_action(self, action_index):
        return self._action_to_malmo_command_array[action_index]

    def step(self, action):
        self._trajectory.append(self._to_malmo_action(action))
        o, r, d, i = super().step(action)

        return o, r, d, i


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 6),
            dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)


# Define a function to create a MineRL environment
def create_env(config):
    mission = config["mission"]
    port = 1000 * config.worker_index + config.vector_index
    print('*********************************************')
    print(f'* Worker {config.worker_index} creating from \
        mission: {mission}, port {port}')
    print('*********************************************')

    if config.worker_index == 0:
        # The first environment is only used for checking the action
        # and observation space. By using a dummy environment, there's
        # no need to spin up a Minecraft instance behind it saving some
        # CPU resources on the head node.
        return DummyEnv()

    env = EnvWrapper(mission, port)
    env = TrackingEnv(env)
    env = FrameStack(env, 2)

    return env


def create_env_for_rollout(config):
    mission = config['mission']
    port = 1000 * config.worker_index + config.vector_index
    print('*********************************************')
    print(f'* Worker {config.worker_index} creating from \
        mission: {mission}, port {port}')
    print('*********************************************')

    env = EnvWrapper(mission, port)
    env = TrackingEnv(env)
    env = FrameStack(env, 2)
    env = TrajectoryWrapper(env)

    return env
