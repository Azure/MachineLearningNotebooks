import traceback
from pyspark.ml.linalg import VectorUDT
from azureml.core.model import Model
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType
from pyspark.sql import SQLContext
from pyspark import SparkContext

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
spark = sqlContext.sparkSession

input_schema = StructType([StructField("features", VectorUDT()), StructField("label", DoubleType())])
reader = spark.read
reader.schema(input_schema)


def init():
    global model
    # note here "iris.model" is the name of the model registered under the workspace
    # this call should return the path to the model.pkl file on the local disk.
    model_path = Model.get_model_path('iris.model')
    # Load the model file back into a LogisticRegression model
    model = LogisticRegressionModel.load(model_path)


def run(data):
    try:
        input_df = reader.json(sc.parallelize([data]))
        result = model.transform(input_df)
        # you can return any datatype as long as it is JSON-serializable
        return result.collect()[0]['prediction']
    except Exception as e:
        traceback.print_exc()
        error = str(e)
        return error
