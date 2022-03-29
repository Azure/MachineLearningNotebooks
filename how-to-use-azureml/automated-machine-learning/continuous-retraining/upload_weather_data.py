import argparse
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import traceback
from azureml.core import Dataset
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace
from azureml.opendatasets import NoaaIsdWeather

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

usaf_list = [
    "725724",
    "722149",
    "723090",
    "722159",
    "723910",
    "720279",
    "725513",
    "725254",
    "726430",
    "720381",
    "723074",
    "726682",
    "725486",
    "727883",
    "723177",
    "722075",
    "723086",
    "724053",
    "725070",
    "722073",
    "726060",
    "725224",
    "725260",
    "724520",
    "720305",
    "724020",
    "726510",
    "725126",
    "722523",
    "703333",
    "722249",
    "722728",
    "725483",
    "722972",
    "724975",
    "742079",
    "727468",
    "722193",
    "725624",
    "722030",
    "726380",
    "720309",
    "722071",
    "720326",
    "725415",
    "724504",
    "725665",
    "725424",
    "725066",
]


def get_noaa_data(start_time, end_time):
    columns = [
        "usaf",
        "wban",
        "datetime",
        "latitude",
        "longitude",
        "elevation",
        "windAngle",
        "windSpeed",
        "temperature",
        "stationName",
        "p_k",
    ]
    isd = NoaaIsdWeather(start_time, end_time, cols=columns)
    noaa_df = isd.to_pandas_dataframe()
    df_filtered = noaa_df[noaa_df["usaf"].isin(usaf_list)]
    df_filtered.reset_index(drop=True)
    print(
        "Received {0} rows of training data between {1} and {2}".format(
            df_filtered.shape[0], start_time, end_time
        )
    )
    return df_filtered


print("Check for new data and prepare the data")

parser = argparse.ArgumentParser("split")
parser.add_argument("--ds_name", help="name of the Dataset to update")
args = parser.parse_args()

print("Argument 1(ds_name): %s" % args.ds_name)

dstor = ws.get_default_datastore()
register_dataset = False
end_time = datetime.utcnow()

try:
    ds = Dataset.get_by_name(ws, args.ds_name)
    end_time_last_slice = ds.data_changed_time.replace(tzinfo=None)
    print("Dataset {0} last updated on {1}".format(args.ds_name, end_time_last_slice))
except Exception:
    print(traceback.format_exc())
    print(
        "Dataset with name {0} not found, registering new dataset.".format(args.ds_name)
    )
    register_dataset = True
    end_time = datetime(2021, 5, 1, 0, 0)
    end_time_last_slice = end_time - relativedelta(weeks=2)

train_df = get_noaa_data(end_time_last_slice, end_time)

if train_df.size > 0:
    print(
        "Received {0} rows of new data after {1}.".format(
            train_df.shape[0], end_time_last_slice
        )
    )
    folder_name = "{}/{:04d}/{:02d}/{:02d}/{:02d}/{:02d}/{:02d}".format(
        args.ds_name,
        end_time.year,
        end_time.month,
        end_time.day,
        end_time.hour,
        end_time.minute,
        end_time.second,
    )
    file_path = "{0}/data.csv".format(folder_name)

    # Add a new partition to the registered dataset
    os.makedirs(folder_name, exist_ok=True)
    train_df.to_csv(file_path, index=False)

    dstor.upload_files(
        files=[file_path], target_path=folder_name, overwrite=True, show_progress=True
    )
else:
    print("No new data since {0}.".format(end_time_last_slice))

if register_dataset:
    ds = Dataset.Tabular.from_delimited_files(
        dstor.path("{}/**/*.csv".format(args.ds_name)),
        partition_format="/{partition_date:yyyy/MM/dd/HH/mm/ss}/data.csv",
    )
    ds.register(ws, name=args.ds_name)
