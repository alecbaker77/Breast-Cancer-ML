from tokenize import Double
import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql import functions as F

conf = SparkConf()
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")

spark = SparkSession.builder.master("local[1]").appName("SparkML").getOrCreate()


#read in CSV as spark dataframe
df = spark.read.format("csv").load("/home/alec/assignment6/breast_cancer.csv", header=True)

#drop the id column as it is not needed
dataset = df.drop("id")

#convert all columns besides diagnosis to double values to work with the vector assembler
for col in dataset.columns:
    if col != "diagnosis":
        dataset = dataset.withColumn(
            col,
            F.col(col).cast(DoubleType())
         )
        
#create a new column which is a String Index of the diagnosis column, this column will be used as our label
indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
dataset = indexer.fit(dataset).transform(dataset)

#since we are using Linear SVC, convert the label to an int so the only two possible values are 0 or 1
dataset = dataset.withColumn("label", F.col("label").cast(IntegerType()))

#remove the diagnosis collumn as it is no longer needed
dataset = dataset.drop("diagnosis")
dataset = dataset.drop("_c32") #dataset had an empty column with null values so remove it

#the features for our vector assembler include the names of all columns
features = dataset.drop("label").columns

#create the vector assembler
vectorAssembler = VectorAssembler(inputCols = features, outputCol='features')

#create vector dataframe to hold column values and label
vectorDF = vectorAssembler.transform(dataset)
vectorDF = vectorDF.select(['features', 'label'])
print("\nEncoded Vector Dataframe")
vectorDF.show(10)

#randomly split the vector dataframe - 80% for testing, 20% for training
splitVectorDF = vectorDF.randomSplit([0.8, 0.2])
testDF = splitVectorDF[0]
trainDF = splitVectorDF[1]


#create the Linear SVC and train it with the train dataframe
linearSVC = LinearSVC(maxIter=50, labelCol="label")
linearSVC = linearSVC.fit(trainDF)

#now test the linearSVC with the test dataframe to get predictions
predictions = linearSVC.transform(testDF)
print("\nPredictions Dataframe")
predictions.show(10)

#use the multiclass classification evaluatore to measure prediction accuracy
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

#print prediction accuracy
print("Prediction Accuracy: " + str(accuracy))