# Custom Heirarchical Time Series Forecasting

## This is a class based solution aimed to solve the problem of heirarchical time series forecasting requiring expensive compute resources and takes a lot of time to train with basic computing resources.

## The idea is that the user specifies their heirarchy combination (state=CA,sku_type=A1) and on run time the dataset would be sliced to have only the data for that combination.Basis which the model will be trained.

## There is a caching layer to avoid retraining of models on the same heirarchy combination.

## Follow the code snippets in Jupyter Notebook to understand how parameteres are to be specified and how training and predction is supposed to be done.

## This might not be exactly accurate(due to run time slicing of dataset),but it's still a good starting point to do hts forecasting with low compute resources.

### [The dataset used here is of Wallmart hiring challenge contest](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
