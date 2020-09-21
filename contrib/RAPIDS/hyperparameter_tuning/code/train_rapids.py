#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import time

import numpy as np
import pandas as pd
import cudf
import cuml

from cuml import RandomForestClassifier as cuRF
from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics.accuracy import accuracy_score

from rapids_csp_azure import RapidsCloudML, PerfTimer
from azureml.core.run import Run

run = Run.get_context()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='location of data')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in RF')
    parser.add_argument('--max_depth', type=int, default=16, help='Max depth of each tree')
    parser.add_argument('--n_bins', type=int, default=8, help='Number of bins used in split point calculation')
    parser.add_argument('--max_features', type=float, default=1.0, help='Number of features for best split')

    args = parser.parse_args()
    data_dir = args.data_dir

    n_estimators = args.n_estimators
    run.log('n_estimators', np.int(args.n_estimators))
    max_depth = args.max_depth
    run.log('max_depth', np.int(args.max_depth))
    n_bins = args.n_bins
    run.log('n_bins', np.int(args.n_bins))
    max_features = args.max_features
    run.log('max_features', np.str(args.max_features))

    print('\n---->>>> cuDF version <<<<----\n', cudf.__version__)
    print('\n---->>>> cuML version <<<<----\n', cuml.__version__)

    compute = 'single-GPU' # 'multi-GPU' option for using multi-GPU algorithms via Dask
    azure_ml = RapidsCloudML(cloud_type='Azure', model_type='RandomForest', data_type='Parquet', compute_type=compute)
    print(compute)

    if compute == 'single-GPU':
        dataset, _ , y_label, _ = azure_ml.load_data(filename=os.path.join(data_dir, 'airline_20m.parquet'))
    else:
       # use parquet files from 'https://airlinedataset.blob.core.windows.net/airline-10years' for multi-GPU training
       dataset, _ , y_label, _ = azure_ml.load_data(filename=os.path.join(data_dir, 'part*.parquet'), 
                                                    col_labels = ['Flight_Number_Reporting_Airline',
                                                                  'Year', 'Quarter', 'Month', 'DayOfWeek',
                                                                  'DOT_ID_Reporting_Airline', 'OriginCityMarketID', 'DestCityMarketID',
                                                                  'DepTime', 'DepDelay', 'DepDel15', 'ArrDel15', 'ArrDelay',
                                                                  'AirTime', 'Distance'], y_label = 'ArrDel15')
    
    X = dataset[dataset.columns.difference(['ArrDelay', y_label])]
    y = dataset[y_label]
    del dataset

    print('\n---->>>> Training using GPUs <<<<----\n')

    # ----------------------------------------------------------------------------------------------------
    # cross-validation folds 
    # ----------------------------------------------------------------------------------------------------
    accuracy_per_fold = []
    train_time_per_fold = []
    infer_time_per_fold = []
    trained_model = []
    global_best_test_accuracy = 0

    model_params = {
        'n_estimators' : n_estimators,
        'max_depth' : max_depth,
        'max_features': max_features,
        'n_bins' : n_bins,
    }

    # optional cross-validation w/ model_params['n_train_folds'] > 1
    for i_train_fold in range(5):
        print( f'\n CV fold { i_train_fold } of { 5 }\n' )

        # split data
        X_train, X_test, y_train, y_test, _ = azure_ml.split_data(X, y, random_state=i_train_fold)
        # train model 
        trained_model, training_time = azure_ml.train_model (X_train, y_train, model_params)

        train_time_per_fold += [ round( training_time, 4) ]

        # evaluate perf
        test_accuracy, infer_time = azure_ml.evaluate_test_perf (trained_model, X_test, y_test)
        accuracy_per_fold += [ round( test_accuracy, 4) ]
        infer_time_per_fold += [ round( infer_time, 4) ]

        # update best model [ assumes maximization of perf metric ]
        if test_accuracy > global_best_test_accuracy :
            global_best_test_accuracy = test_accuracy

    run.log('Total training inference time', np.float(training_time + infer_time))
    run.log('Accuracy', np.float(global_best_test_accuracy))
    print( '\n Accuracy             :', global_best_test_accuracy)
    print( '\n accuracy per fold    :', accuracy_per_fold)
    print( '\n train-time per fold  :', train_time_per_fold)
    print( '\n train-time all folds  :', sum(train_time_per_fold))
    print( '\n infer-time per fold  :', infer_time_per_fold)
    print( '\n infer-time all folds  :', sum(infer_time_per_fold))

if __name__ == '__main__':
    with PerfTimer() as total_script_time:
        main()
    print('Total runtime: {:.2f}'.format(total_script_time.duration))
    run.log('Total runtime', np.float(total_script_time.duration))
    print('\n Exiting script')
