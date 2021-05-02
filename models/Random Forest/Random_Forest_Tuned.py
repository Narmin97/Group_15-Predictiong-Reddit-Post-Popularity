"""
Random Forest Tuned Regression.
"""
from __future__ import print_function


from pyspark import SparkContext, SQLContext

from functools import reduce
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


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
    
    rfparamGrid = (ParamGridBuilder().addGrid(rf.maxDepth, [5, 10, 15, 20, 25, 30])
             .addGrid(rf.maxBins, [25, 35, 45, 55, 65])
             .build())

    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    
    rfcv = CrossValidator(estimator = pipeline,
                      estimatorParamMaps = rfparamGrid,
                      evaluator = evaluator,
                      numFolds = 5)
    
    rfcvModel = rfcv.fit(trainingData)
    print(rfcvModel)

    rfpredictions = rfcvModel.transform(testData)
    print('Accuracy:', evaluator.evaluate(rfpredictions))