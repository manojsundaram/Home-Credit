from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

spark.sql("select * from staging.orders")