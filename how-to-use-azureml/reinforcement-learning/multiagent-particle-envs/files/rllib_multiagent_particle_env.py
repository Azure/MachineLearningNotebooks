# Some code taken from: https://github.com/wsjeon/maddpg-rllib/

import imp
import os

import gym
from gym import wrappers
from ray import rllib

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


CUSTOM_SCENARIOS = ['simple_switch']


class ParticleEnvRenderWrapper(gym.Wrapper):
    def __init__(self, env, horizon):
        super().__init__(env)
        self.horizon = horizon

    def reset(self):
        self._num_steps = 0

        return self.env.reset()

    def render(self, mode):
        if mode == 'human':
            self.env.render(mode=mode)
        else:
            return self.env.render(mode=mode)[0]

    def step(self, actions):
        obs_list, rew_list, done_list, info_list = self.env.step(actions)

        self._num_steps += 1
        done = (all(done_list) or self._num_steps >= self.horizon)

        # Gym monitor expects reward to be an int.  This is only used for its
        # stats reporter, which we're not interested in.  To make video recording
        # work, we package the rewards in the info object and extract it below.
        return obs_list, 0, done, [rew_list, done_list, info_list]


class RLlibMultiAgentParticleEnv(rllib.MultiAgentEnv):
    def __init__(self, scenario_name, horizon, monitor_enabled=False, video_frequency=500):
        self._env = _make_env(scenario_name, horizon, monitor_enabled, video_frequency)
        self.num_agents = self._env.n
        self.agent_ids = list(range(self.num_agents))

        self.observation_space_dict = self._make_dict(self._env.observation_space)
        self.action_space_dict = self._make_dict(self._env.action_space)

    def reset(self):
        obs_dict = self._make_dict(self._env.reset())
        return obs_dict

    def step(self, action_dict):
        actions = list(action_dict.values())
        obs_list, _, _, infos = self._env.step(actions)
        rew_list, done_list, _ = infos

        obs_dict = self._make_dict(obs_list)
        rew_dict = self._make_dict(rew_list)
        done_dict = self._make_dict(done_list)
        done_dict['__all__'] = all(done_list)
        info_dict = self._make_dict([{'done': done} for done in done_list])

        return obs_dict, rew_dict, done_dict, info_dict

    def render(self, mode='human'):
        self._env.render(mode=mode)

    def _make_dict(self, values):
        return dict(zip(self.agent_ids, values))


def _video_callable(video_frequency):
    def should_record_video(episode_id):
        if episode_id % video_frequency == 0:
            return True
        return False

    return should_record_video


def _make_env(scenario_name, horizon, monitor_enabled, video_frequency):
    if scenario_name in CUSTOM_SCENARIOS:
        # Scenario file must exist locally
        file_path = os.path.join(os.path.dirname(__file__), scenario_name + '.py')
        scenario = imp.load_source('', file_path).Scenario()
    else:
        scenario = scenarios.load(scenario_name + '.py').Scenario()

    world = scenario.make_world()

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.metadata['video.frames_per_second'] = 8

    env = ParticleEnvRenderWrapper(env, horizon)

    if not monitor_enabled:
        return env

    return wrappers.Monitor(env, './logs/videos', resume=True, video_callable=_video_callable(video_frequency))


def env_creator(config):
    monitor_enabled = False
    if hasattr(config, 'worker_index') and hasattr(config, 'vector_index'):
        monitor_enabled = (config.worker_index == 1 and config.vector_index == 0)

    return RLlibMultiAgentParticleEnv(**config, monitor_enabled=monitor_enabled)
