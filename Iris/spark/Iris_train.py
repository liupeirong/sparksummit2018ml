# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC This is a multi-class classification problem.  The model predicts 3 different types of Iris flowers. 

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark
import os
import urllib
import sys

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Iris dataset
# MAGIC You can download Iris data from the internet, for example, from [uci web site](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/).

# COMMAND ----------

schema = StructType([
  StructField('sepal-length', FloatType()),
  StructField('sepal-width', FloatType()),
  StructField('petal-length', FloatType()),
  StructField('petal-width', FloatType()),
  StructField('class', StringType()),
])

df = spark.read.format('csv').option('header', 'false').option('sep', ',').schema(schema).load('/mnt/mldata/iris.csv')
  
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the feature extraction and model training pipeline
# MAGIC * Vectorize all numeric feature columns into a single features vector
# MAGIC * Convert text labels to numeric indices
# MAGIC * Create a logistic regression model

# COMMAND ----------

feature_cols = df.columns[:-1]
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
label_indexer = StringIndexer(inputCol='class', outputCol='label')
lr = LogisticRegression(family='multinomial', elasticNetParam=0.2)

pipeline = Pipeline(stages=[assembler, label_indexer, lr])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split data for training and testing, train the model

# COMMAND ----------

train, test = df.randomSplit([0.7, 0.3], 42)
model = pipeline.fit(train)
print(model.stages[2].coefficientMatrix)
print(model.stages[2].interceptVector)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the model

# COMMAND ----------

prediction = model.transform(test)
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
accuracy = evaluator.evaluate(prediction)
accuracy