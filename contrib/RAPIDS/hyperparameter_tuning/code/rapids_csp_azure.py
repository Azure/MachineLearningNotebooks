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

import json
import logging
import os
import pprint
import sys
import time

import dask
import numpy as np
import pandas as pd
import psutil
import sklearn
from dask.distributed import Client, wait
from sklearn import ensemble
from sklearn.model_selection import \
    train_test_split as sklearn_train_test_split

import cudf
import cuml
import cupy
import dask_cudf
import pynvml
import xgboost
from cuml.dask.common import utils as dask_utils

from cuml.dask.ensemble import RandomForestClassifier as cumlDaskRF
from cuml.metrics.accuracy import accuracy_score
from cuml.preprocessing.model_selection import \
    train_test_split as cuml_train_test_split
from dask_cuda import LocalCUDACluster
from dask_ml.model_selection import train_test_split as dask_train_test_split

default_azureml_paths = {
    'train_script' : './train_script',
    'train_data' : './data_airline',
    'output' : './output',
}

class RapidsCloudML(object):

    def __init__(self, cloud_type = 'Azure', 
                   model_type = 'RandomForest', 
                   data_type = 'Parquet',
                   compute_type = 'single-GPU', 
                   verbose_estimator = False,
                   CSP_paths = default_azureml_paths):

        self.CSP_paths = CSP_paths
        self.cloud_type = cloud_type
        self.model_type = model_type
        self.data_type = data_type
        self.compute_type = compute_type
        self.verbose_estimator = verbose_estimator
        self.log_to_file(f'\n> RapidsCloudML\n\tCompute, Data , Model, Cloud types {self.compute_type, self.data_type, self.model_type, self.cloud_type}')

        # Setting up client for multi-GPU option
        if 'multi' in self.compute_type:
            self.log_to_file("\n\tMulti-GPU selected")
            # This will use all GPUs on the local host by default
            cluster = LocalCUDACluster(threads_per_worker=1)
            self.client = Client(cluster)

            # Query the client for all connected workers
            self.workers = self.client.has_what().keys()
            self.n_workers = len(self.workers)
            self.log_to_file(f'\n\tClient information {self.client}')

    def load_hyperparams(self, model_name = 'XGBoost'):
        """
        Selecting model paramters based on the model we select for execution.
        Checks if there is a config file present in the path self.CSP_paths['hyperparams'] with
        the parameters for the experiment. If not present, it returns the default parameters.

        Parameters
        ----------
        model_name : string
                     Selects which model to set the parameters for. Takes either 'XGBoost' or 'RandomForest'.

        Returns
        ----------
        model_params : dict
                       Loaded model parameters (dict)
        """

        self.log_to_file('\n> Loading Hyperparameters')

        # Default parameters of the models
        if self.model_type == 'XGBoost':
            # https://xgboost.readthedocs.io/en/latest/parameter.html
            model_params = { 
                'max_depth': 6,
                'num_boost_round': 100, 
                'learning_rate': 0.3,
                'gamma': 0.,
                'lambda': 1.,
                'alpha': 0.,
                'objective':'binary:logistic',
                'random_state' : 0
            }
            
        elif self.model_type == 'RandomForest':
            # https://docs.rapids.ai/api/cuml/stable/  -> cuml.ensemble.RandomForestClassifier
            model_params = {
                'n_estimators' : 10,
                'max_depth' : 10,
                'n_bins' : 16,
                'max_features': 1.0,
                'seed' : 0,
            }

        hyperparameters = {}
        try:
            with open(self.CSP_paths['hyperparams'], 'r') as file_handle:
                hyperparameters = json.load(file_handle)
                for key, value in hyperparameters.items():
                    model_params[key] = value
                pprint.pprint(model_params)
                return model_params

        except Exception as error:
            self.log_to_file(str(error))
            return

    def load_data(self, filename = 'dataset.orc', col_labels = None, y_label = 'ArrDelayBinary'):
        """
        Loading the data into the object from the filename and based on the columns that we are
        interested in. Also, generates y_label from 'ArrDelay' column to convert this into a binary
        classification problem.

        Parameters
        ----------
        filename : string
                   the path of the dataset to be loaded

        col_labels : list of strings
                     The input columns that we are interested in. None selects all the columns

        y_label : string
                  The column to perform the prediction task in.

        Returns
        ----------
        dataset : dataframe (Pandas, cudf or dask-cudf)
                  Ingested dataset in the format of a dataframe

        col_labels : list of strings
                     The input columns selected

        y_label : string
                  The generated y_label name for binary classification

        duration : float
                   The time it took to execute the function
        """
        target_filename = filename
        self.log_to_file( f'\n> Loading dataset from {target_filename}')

        with PerfTimer() as ingestion_timer:
            if 'CPU' in self.compute_type:
                # CPU Reading options
                self.log_to_file(f'\n\tCPU read')

                if self.data_type == 'ORC':
                    with open( target_filename, mode='rb') as file:
                        dataset = pyarrow_orc.ORCFile(file).read().to_pandas()
                elif self.data_type == 'CSV':
                    dataset = pd.read_csv( target_filename, names = col_labels )
                    
                elif self.data_type == 'Parquet':
                    
                    if 'single' in self.compute_type:
                        dataset = pd.read_parquet(target_filename)
                    
                    elif 'multi' in self.compute_type:
                        self.log_to_file(f'\n\tReading using dask dataframe')
                        dataset = dask.dataframe.read_parquet(target_filename, columns = columns)

            elif 'GPU' in self.compute_type:
                # GPU Reading Option

                self.log_to_file(f'\n\tGPU read')
                if self.data_type == 'ORC':
                    dataset = cudf.read_orc(target_filename)

                elif self.data_type == 'CSV':
                    dataset = cudf.read_csv(target_filename, names = col_labels)

                elif self.data_type == 'Parquet':

                    if 'single' in self.compute_type:
                        dataset = cudf.read_parquet(target_filename)

                    elif 'multi' in self.compute_type:
                        self.log_to_file(f'\n\tReading using dask_cudf')
                        dataset = dask_cudf.read_parquet(target_filename, columns = col_labels)

        # cast all columns to float32
        for col in dataset.columns:
            dataset[col] = dataset[col].astype(np.float32)  # needed for random forest

        # Adding y_label column if it is not present
        if y_label not in dataset.columns:
            dataset[y_label] = 1.0 * (
                    dataset["ArrDelay"] > 10
                )

        dataset[y_label] = dataset[y_label].astype(np.int32) # Needed for cuml RF
        
        dataset = dataset.fillna(0.0) # Filling the null values. Needed for dask-cudf

        self.log_to_file(f'\n\tIngestion completed in {ingestion_timer.duration}')
        self.log_to_file(f'\n\tDataset descriptors: {dataset.shape}\n\t{dataset.dtypes}')
        return dataset, col_labels, y_label, ingestion_timer.duration

    def split_data(self, dataset, y_label, train_size = .8, random_state = 0, shuffle = True):
        """
        Splitting data into train and test split, has appropriate imports for different compute modes.
        CPU compute - Uses sklearn, we manually filter y_label column in the split call
        GPU Compute - Single GPU uses cuml and multi GPU uses dask, both split y_label internally.

        Parameters
        ----------
        dataset : dataframe
                  The dataframe on which we wish to perform the split
        y_label : string
                  The name of the column (not the series itself)
        train_size : float
                     The size for the split. Takes values between 0 to 1.
        random_state : int
                       Useful for running reproducible splits.
        shuffle : binary
                  Specifies if the data must be shuffled before splitting.

        Returns
        ----------
        X_train : dataframe
                  The data to be used for training. Has same type as input dataset.
        X_test : dataframe
                  The data to be used for testing. Has same type as input dataset.
        y_train : dataframe
                  The label to be used for training. Has same type as input dataset.
        y_test : dataframe
                  The label to be used for testing. Has same type as input dataset.
        duration : float
                   The time it took to perform the split
        """
        self.log_to_file('\n> Splitting train and test data')
        start_time = time.perf_counter()

        with PerfTimer() as split_timer:
            if 'CPU' in self.compute_type:
                X_train, X_test, y_train, y_test = sklearn_train_test_split(dataset.loc[:, dataset.columns != y_label],
                                                                            dataset[y_label],
                                                                            train_size = train_size,
                                                                            shuffle = shuffle,
                                                                            random_state = random_state)

            elif 'GPU' in self.compute_type:
                if 'single' in self.compute_type:
                    X_train, X_test, y_train, y_test = cuml_train_test_split(X = dataset,
                                                                             y = y_label,
                                                                             train_size = train_size,
                                                                             shuffle = shuffle,
                                                                             random_state = random_state) 
                elif 'multi' in self.compute_type:
                    X_train, X_test, y_train, y_test = dask_train_test_split(dataset,
                                                                             y_label,
                                                                             train_size = train_size,
                                                                             shuffle = False, # shuffle not available for dask_cudf yet
                                                                             random_state = random_state)
        
        self.log_to_file(f'\n\tX_train shape and type{X_train.shape} {type(X_train)}')
        self.log_to_file( f'\n\tSplit completed in {split_timer.duration}')
        return X_train, X_test, y_train, y_test, split_timer.duration

    def train_model(self, X_train, y_train, model_params):
        """
        Trains a model with the model_params specified by calling fit_xgboost or
        fit_random_forest depending on the model_type.

        Parameters
        ----------
        X_train : dataframe
                  The data for traning
        y_train : dataframe
                  The label to be used for training.
        model_params : dict
                       The model params to use for this training
        Returns
        ----------
        trained_model : The object of the trained model either of XGBoost or RandomForest

        training_time : float
                        The time it took to train the model
        """
        self.log_to_file(f'\n> Training {self.model_type} estimator w/ hyper-params')
        training_time = 0

        try:
            if self.model_type == 'XGBoost':
                trained_model, training_time = self.fit_xgboost(X_train, y_train, model_params)
            elif self.model_type == 'RandomForest':
                trained_model, training_time = self.fit_random_forest(X_train, y_train, model_params)
        except Exception as error:
            self.log_to_file('\n\n!error during model training: ' + str(error))
        self.log_to_file( f'\n\tFinished training in {training_time:.4f} s')
        return trained_model, training_time

    def fit_xgboost(self, X_train, y_train, model_params):
        """
        Trains a XGBoost model on X_train and y_train with model_params

        Parameters and Objects returned are same as trained_model
        """             
        if 'GPU' in self.compute_type:
            model_params.update({'tree_method': 'gpu_hist'})
        else:
            model_params.update({'tree_method': 'hist'})
        
        with PerfTimer() as train_timer:
            if 'single' in self.compute_type:
                train_DMatrix = xgboost.DMatrix(data = X_train, label = y_train)
                trained_model = xgboost.train(dtrain = train_DMatrix,
                                              params = model_params,
                                              num_boost_round = model_params['num_boost_round'])
            elif 'multi' in self.compute_type:
                self.log_to_file("\n\tTraining multi-GPU XGBoost")
                train_DMatrix = xgboost.dask.DaskDMatrix(self.client, data = X_train, label = y_train)
                trained_model = xgboost.dask.train(self.client,
                                                   dtrain = train_DMatrix,
                                                   params = model_params,
                                                   num_boost_round = model_params['num_boost_round'])
        return trained_model, train_timer.duration

    def fit_random_forest ( self, X_train, y_train, model_params ):
        """
        Trains a RandomForest model on X_train and y_train with model_params.
        Depending on compute_type, estimators from appropriate packages are used.
        CPU - sklearn
        Single-GPU - cuml
        multi_gpu - cuml.dask

        Parameters and Objects returned are same as trained_model
        """
        if 'CPU' in self.compute_type:
            rf_model = sklearn.ensemble.RandomForestClassifier(n_estimators = model_params['n_estimators'],
                                                                max_depth = model_params['max_depth'],
                                                                max_features = model_params['max_features'], 
                                                                n_jobs = int(self.n_workers),
                                                                verbose = self.verbose_estimator)
        elif 'GPU' in self.compute_type:
            if 'single' in self.compute_type:
                rf_model = cuml.ensemble.RandomForestClassifier(n_estimators = model_params['n_estimators'],
                                                                max_depth = model_params['max_depth'],
                                                                n_bins = model_params['n_bins'],
                                                                max_features = model_params['max_features'],
                                                                verbose = self.verbose_estimator)
            elif 'multi' in self.compute_type:
                self.log_to_file("\n\tFitting multi-GPU daskRF")
                X_train, y_train = dask_utils.persist_across_workers(self.client,
                                                                     [X_train.fillna(0.0),
                                                                     y_train.fillna(0.0)],
                                                                     workers=self.workers)
                rf_model = cuml.dask.ensemble.RandomForestClassifier(n_estimators = model_params['n_estimators'],
                                                                       max_depth = model_params['max_depth'],
                                                                       n_bins = model_params['n_bins'],
                                                                       max_features = model_params['max_features'],
                                                                       verbose = self.verbose_estimator)
        with PerfTimer() as train_timer:
            try:
                trained_model = rf_model.fit( X_train, y_train)
            except Exception as error:
                self.log_to_file( "\n\n! Error during fit " + str(error))
        return trained_model, train_timer.duration
    
    def evaluate_test_perf(self, trained_model, X_test, y_test, threshold=0.5):
        """
        Evaluates the model performance on the inference set. For XGBoost we need
        to generate a DMatrix and then we can evaluate the model.
        For Random Forest, in single GPU case, we can just call .score function.
        And multi-GPU Random Forest needs to predict on the model and then compute
        the accuracy score.

        Parameters
        ----------
        trained_model : The object of the trained model either of XGBoost or RandomForest
        X_test : dataframe
                  The data for testing
        y_test : dataframe
                  The label to be used for testing.
        Returns
        ----------
        test_accuracy : float
                        The accuracy achieved on test set
        duration : float
                   The time it took to evaluate the model
        """
        self.log_to_file(f'\n> Inferencing on test set')
        test_accuracy = None
        with PerfTimer() as inference_timer:
            try:
                if self.model_type == 'XGBoost':
                    if 'multi' in self.compute_type:
                        test_DMatrix = xgboost.dask.DaskDMatrix(self.client, data = X_test, label = y_test)
                        xgb_pred = xgboost.dask.predict(self.client, trained_model, test_DMatrix).compute()
                        xgb_pred = (xgb_pred > threshold) * 1.0
                        test_accuracy = accuracy_score(y_test.compute(), xgb_pred)
                    elif 'single' in self.compute_type: 
                        test_DMatrix = xgboost.DMatrix(data = X_test, label = y_test)
                        xgb_pred = trained_model.predict(test_DMatrix)
                        xgb_pred = (xgb_pred > threshold) * 1.0
                        test_accuracy = accuracy_score(y_test, xgb_pred)

                elif self.model_type == 'RandomForest':
                    if 'multi' in self.compute_type:
                        cuml_pred = trained_model.predict(X_test).compute()
                        self.log_to_file("\n\tPrediction complete")
                        test_accuracy = accuracy_score(y_test.compute(), cuml_pred, convert_dtype=True)
                    elif 'single' in self.compute_type:
                        test_accuracy = trained_model.score( X_test, y_test.astype('int32') )

            except Exception as error:
                self.log_to_file( '\n\n!error during inference: ' + str(error))

        self.log_to_file(f'\n\tFinished inference in {inference_timer.duration:.4f} s')
        self.log_to_file(f'\n\tTest-accuracy: {test_accuracy}')
        return test_accuracy, inference_timer.duration

    def set_up_logging( self ):
        """
        Function to set up logging for the object.
        """
        logging_path = self.CSP_paths['output'] + '/log.txt'
        logging.basicConfig( filename= logging_path,
                             level=logging.INFO)

    def log_to_file ( self, text ):
        """
        Logs the text that comes in as input.
        """
        logging.info( text )
        print(text)

# perf_counter = highest available timer resolution 
class PerfTimer:
    def __init__(self):
        self.start = None
        self.duration = None
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.duration = time.perf_counter() - self.start