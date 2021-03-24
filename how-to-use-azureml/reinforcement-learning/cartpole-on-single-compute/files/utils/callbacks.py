'''RLlib callbacks module:
    Common callback methods to be passed to RLlib trainer.
'''

from azureml.core import Run


def on_train_result(info):
    '''Callback on train result to record metrics returned by trainer.
    '''
    run = Run.get_context().parent
    run.log(
        name='episode_reward_mean',
        value=info["result"]["episode_reward_mean"])
    run.log(
        name='episodes_total',
        value=info["result"]["episodes_total"])
    run.log(
        name='perf_cpu_percent',
        value=info["result"]["perf"]["cpu_util_percent"])
    run.log(
        name='perf_memory_percent',
        value=info["result"]["perf"]["ram_util_percent"])
