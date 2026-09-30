from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Test").master("spark://spark-master:7077").getOrCreate()
print("========================================")
print("SPARK CLUSTER CONNECTED SUCCESSFULLY!")
print("Spark version:", spark.version)
print("Master URL:", spark.sparkContext.master)
print("========================================")
spark.stop()
