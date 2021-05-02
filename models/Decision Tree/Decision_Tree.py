"""
Decision Tree Regression.
"""
from __future__ import print_function

from pyspark import SparkContext, SQLContext

from functools import reduce
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    
    sc = SparkContext(appName="decision_tree_classification_example")
    sqlContext = SQLContext(sc)

    # Load the data 
    data = sqlContext.read.format('csv').option('header', False).option('multiline', True).load('hdfs://master:9000/user/hadoop/output/output-20/part-00000')
    
    data = data.drop(*["_c0", "_c2", "_c3", "_c5", "_c6", "_c7", "_c9", "_c10","_c12"])
    
    oldColumns = data.schema.names
    newColumns = ["utc", "subreddit", "sentiment", "score"]
    data = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), data)
    
   
    # Changing from utc seconds to local time
    data = data.withColumn("created_utc", f.from_unixtime("created_utc", "dd/MM/yyyy HH:MM:SS"))
    data = data.withColumn("utc",f.to_timestamp(data["utc"], format='MM/dd/yyyy HH:mm:ss'))
    # Creating hour od day column
    data = data.withColumn("HoD", f.date_format(data["utc"], "H"))
    # Creating day of week column
    data = data.withColumn("DoW", f.date_format(data["utc"], "EEEE"))
    # Creating month column
    data = data.withColumn("month", f.date_format(data["utc"], "M"))
    
    data = data.drop(data["utc"])
    
    # Convert the score variable from string to int
    data = data.withColumn("score",data.score.cast(IntegerType()))
    
    #create a list of the columns that are string typed
    categoricalColumns = [item[0] for item in data.dtypes if item[1].startswith('string') ]
    
    
    #define a list of stages in the pipeline. The string indexer will be one stage
    stages = []

    #iterate through all categorical values
    for categoricalCol in categoricalColumns:
        #create a string indexer for those categorical values and assign a new name including the word 'Index'
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')

        #append the string Indexer to our list of stages
        stages += [stringIndexer]
        
    #Create the pipeline. Assign the stages list to the pipeline key word stages
    pipeline = Pipeline(stages = stages)
    #fit the pipeline to our dataframe
    pipelineModel = pipeline.fit(data)
    #transform the dataframe
    data = pipelineModel.transform(data)
    data = data.drop(*categoricalColumns)
    
    assembler = VectorAssembler( inputCols=['subredditIndex', 'sentimentIndex', 'HoDIndex', 'DoWIndex', 'monthIndex'], 
                                outputCol="features")
    
    output = assembler.transform(data)
    output = output.withColumn('label', output['score'])
    data = output.select("features", "label")

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 25 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=25).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

    # Chain indexer and tree in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, dt])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    treeModel = model.stages[1]
    # summary only
    print(treeModel)
