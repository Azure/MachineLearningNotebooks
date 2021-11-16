# Custom Heirarchical Time Series Forecasting

## Problem with conventional Time Series Forecasting :

### Using a standard compute resource (STANDARD_DS11_V2) and traning the model over the dataset provided in the Data folder,took more than 3 hours to train the model.As per the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-forecast), I am assuming that the model is trained on all heirarchy combinations of the data.<br><br>

### The issue is that I wanted forecasts for a very niche set of query i.e state=WA,store_id=1,product_category=B,SKU=B2,but I had to wait for the whole traning to get over hence I have come up with this class-based solution to address this issue by natively using Azure Automl <br><br>

### What really happens is that I dynamically slice the dataset basis the heirarchy variables the user passes upon which it requires the forecast for,then using the sliced dataset I train the model and give out predictions for the same.There is an additional caching feature to avoid retraining of models on the same heirarchy variables.

### This is benificial in 2 ways:<br>

- ### We can work with the standard compute resource and don't need to acquire more expensive compute resources
- ### Traning time reduced significantly, as we train only on what's needed at this point in time.

### This might not be exactly accurate(due to run time slicing of dataset),but it's still a good starting point to do hts forecasting with low compute resources.
