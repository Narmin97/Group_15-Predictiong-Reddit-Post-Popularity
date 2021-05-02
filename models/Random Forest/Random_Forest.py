"""
Random Forest Regression.
"""
from __future__ import print_function


from pyspark import SparkContext, SQLContext

from functools import reduce
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator


if __name__ == "__main__":
    
    sc = SparkContext(appName="decision_tree_classification_example")
    sqlContext = SQLContext(sc)

    # Loading and Preparing the data
    data = sqlContext.read.format('csv').option('header', False).option('multiline', True).load('hdfs://master:9000/user/hadoop/output/output-20/part-00000')
    data = data.drop(*["_c0", "_c2", "_c3", "_c7", "_c9", "_c12"])
    
    oldColumns = data.schema.names
    newColumns = ["subreddit", "gilded", "archived", "sentiment", "controversiality", "score", "HoD", 
                  "DoW", "DoM", "month", "year"]
    data = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), data)
    
    data = data.withColumn("score",data.score.cast(IntegerType()))
    
    categoricalColumns = [item[0] for item in data.dtypes if item[1].startswith('string') ]
    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        stages += [stringIndexer]
        
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(data)
    data = pipelineModel.transform(data)
    data = data.drop(*categoricalColumns)
    
    assembler = VectorAssembler( inputCols=['subredditIndex', 'gildedIndex', 'archivedIndex', 'sentimentIndex', 
                                            'controversialityIndex', 'HoDIndex', 'DoWIndex', 'DoMIndex', 'monthIndex',
                                            'yearIndex'], 
                                outputCol="features")
    
    output = assembler.transform(data)
    output = output.withColumn('label', output['score'])
    data = output.select("features", "label")

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=25).fit(data)

    # Training the model and getting the accuracy    
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    rf = RandomForestRegressor(featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[featureIndexer, rf])
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)


    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    treeModel = model.stages[1]
    print(treeModel)

