import json
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import Ridge

from azureml.core.model import Model
import azureml.train.automl


def init():
    global models
    models = {}
    global group_columns_str
    group_columns_str = "<<groups>>"
    global time_column_name
    time_column_name = "<<time_colname>>"

    global group_columns
    group_columns = group_columns_str.split("#####")
    global valid_chars
    valid_chars = re.compile('[^a-zA-Z0-9-]')


def run(raw_data):
    try:
        data = pd.read_json(raw_data)
        # Make sure we have correct time points.
        data[time_column_name] = pd.to_datetime(data[time_column_name], unit='ms')
        dfs = []
        for grain, df_one in data.groupby(group_columns):
            if isinstance(grain, int):
                cur_group = str(grain)
            elif isinstance(grain, str):
                cur_group = grain
            else:
                cur_group = "#####".join([str(v) for v in list(grain)])
            cur_group = valid_chars.sub('', cur_group)
            print("Query model for group {}".format(cur_group))
            if cur_group not in models:
                model_path = Model.get_model_path(cur_group)
                model = joblib.load(model_path)
                models[cur_group] = model
            _, xtrans = models[cur_group].forecast(df_one)
            dfs.append(xtrans)
        df_ret = pd.concat(dfs)
        df_ret.reset_index(drop=False, inplace=True)
        return json.dumps({'predictions': df_ret.to_json()})

    except Exception as e:
        error = str(e)
        return error
