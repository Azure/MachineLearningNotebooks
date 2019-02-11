# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import pyspark
import os
import urllib
import sys

from pyspark.sql.functions import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

from azureml.core.run import Run

# initialize logger
run = Run.get_context()

# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('Iris').getOrCreate()

# print runtime versions
print('****************')
print('Python version: {}'.format(sys.version))
print('Spark version: {}'.format(spark.version))
print('****************')

# load iris.csv into Spark dataframe
schema = StructType([
    StructField("sepal-length", DoubleType()),
    StructField("sepal-width", DoubleType()),
    StructField("petal-length", DoubleType()),
    StructField("petal-width", DoubleType()),
    StructField("class", StringType())
])

data = spark.read.format("com.databricks.spark.csv") \
    .option("header", "true") \
    .schema(schema) \
    .load("iris.csv")

print("First 10 rows of Iris dataset:")
data.show(10)

# vectorize all numerical columns into a single feature column
feature_cols = data.columns[:-1]
assembler = pyspark.ml.feature.VectorAssembler(
    inputCols=feature_cols, outputCol='features')
data = assembler.transform(data)

# convert text labels into indices
data = data.select(['features', 'class'])
label_indexer = pyspark.ml.feature.StringIndexer(
    inputCol='class', outputCol='label').fit(data)
data = label_indexer.transform(data)

# only select the features and label column
data = data.select(['features', 'label'])
print("Reading for machine learning")
data.show(10)

# change regularization rate and you will likely get a different accuracy.
reg = 0.01
# load regularization rate from argument if present
if len(sys.argv) > 1:
    reg = float(sys.argv[1])

# log regularization rate
run.log("Regularization Rate", reg)

# use Logistic Regression to train on the training set
train, test = data.randomSplit([0.70, 0.30])
lr = pyspark.ml.classification.LogisticRegression(regParam=reg)
model = lr.fit(train)

# predict on the test set
prediction = model.transform(test)
print("Prediction")
prediction.show(10)

# evaluate the accuracy of the model using the test set
evaluator = pyspark.ml.evaluation.MulticlassClassificationEvaluator(
    metricName='accuracy')
accuracy = evaluator.evaluate(prediction)

print()
print('#####################################')
print('Regularization rate is {}'.format(reg))
print("Accuracy is {}".format(accuracy))
print('#####################################')
print()

# log accuracy
run.log('Accuracy', accuracy)
