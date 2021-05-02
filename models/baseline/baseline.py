"""
Baseline Regression.
"""

from __future__ import print_function

from pyspark import SparkContext, SQLContext

from math import sqrt
from functools import reduce
from pyspark.sql.types import IntegerType

from pyspark.sql.functions import lit

if __name__ == "__main__":
    
    sc = SparkContext(appName="baseline-model")
    sqlContext = SQLContext(sc)
    
    # Loading and Preparing the data
    data = sqlContext.read.format('csv').option('header', False).option('multiline', True).load('hdfs://master:9000/user/hadoop/output/output-20/part-00000')
    data = data.drop(*["_c0", "_c2", "_c3", "_c7", "_c9", "_c12"])
    
    oldColumns = data.schema.names
    newColumns = ["subreddit", "gilded", "archived", "sentiment", "controversiality", "score", "HoD", 
                  "DoW", "DoM", "month", "year"]
    data = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), data)
    
    data = data.withColumn("score",data.score.cast(IntegerType()))
    
    # Training the model and getting the accuracy
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    mean_score = float(trainingData.select('score').groupBy().sum().first()[0]) / trainingData.count()
    testData_withPreds = testData.withColumn('prediction', lit(mean_score))
    
    def calcRMSE(data, labelCol='score'):
         return sqrt(float(data.rdd.map(lambda row: (row[labelCol]-row['prediction'])**2).reduce(lambda a,b: a+b)) / data.count())

    
    print ('RMSE: %4.2f' % (calcRMSE(testData_withPreds,labelCol='score')))


    
    