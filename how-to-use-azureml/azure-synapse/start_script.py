from pyspark.sql import SparkSession

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="")
parser.add_argument("--output", default="")

args, unparsed = parser.parse_known_args()

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

arr = sc._gateway.new_array(sc._jvm.java.lang.String, 2)
arr[0] = args.input
arr[1] = args.output

obj = sc._jvm.WordCount
obj.main(arr)
