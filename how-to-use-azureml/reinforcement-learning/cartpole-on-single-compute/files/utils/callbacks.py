'''RLlib callbacks module:
    Common callback methods to be passed to RLlib trainer.
'''

from azureml.core import Run
from ray import tune
from ray.tune import Callback
from ray.air import session


class TrialCallback(Callback):

    def on_trial_result(self, iteration, trials, trial, result, **info):
        '''Callback on train result to record metrics returned by trainer.
        '''
        run = Run.get_context()
        run.log(
            name='episode_reward_mean',
            value=result["episode_reward_mean"])
        run.log(
            name='episodes_total',
            value=result["episodes_total"])
