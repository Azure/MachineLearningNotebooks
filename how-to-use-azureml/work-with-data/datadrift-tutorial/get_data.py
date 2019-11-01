# import packages
import os
import pandas as pd
from calendar import monthrange
from datetime import datetime, timedelta
from azureml.core import Dataset, Datastore, Workspace
from azureml.opendatasets import NoaaIsdWeather

# get workspace and datastore
ws = Workspace.from_config()
dstore = ws.get_default_datastore()

# adjust parameters as needed
target_years = list(range(2010, 2020))
start_month = 1

# get data
for year in target_years:
    for month in range(start_month, 12 + 1):
        path = 'weather-data/{}/{:02d}/'.format(year, month)
        try:
            start = datetime(year, month, 1)
            end = datetime(year, month, monthrange(year, month)[1]) + timedelta(days=1)
            isd = NoaaIsdWeather(start, end).to_pandas_dataframe()
            isd = isd[isd['stationName'].str.contains('FLORIDA', regex=True, na=False)]
            os.makedirs(path, exist_ok=True)
            isd.to_parquet(path + 'data.parquet')
        except Exception as e:
            print('Month {} in year {} likely has no data.\n'.format(month, year))
            print('Exception: {}'.format(e))
