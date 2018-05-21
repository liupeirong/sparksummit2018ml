// Databricks notebook source
// MAGIC %md
// MAGIC ## This example is based on Microsoft Machine Learning example [Churn Prediction](https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/scenario-churn-prediction)
// MAGIC This is a binary classification problem. Download data [here](https://github.com/Azure/MachineLearningSamples-ChurnPrediction/tree/master/data). 

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._
import org.apache.spark.sql.types._

// COMMAND ----------

// MAGIC %md
// MAGIC ## Read and clean raw data

// COMMAND ----------

val df = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("/mnt/mldata/CATelcoCustomerChurnTrainingSample.csv")
    .dropDuplicates()
    .drop("year")
    .drop("month")
    .na.fill(0)
display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Build the machine learning pipeline
// MAGIC * Convert text features to numerice indices
// MAGIC * Combine feature columns into a single vector column of features
// MAGIC * Create the decision tree classifier

// COMMAND ----------

val categorical_columns = df.dtypes.filter(_._2 == "StringType").map(_._1) //TODO: non-numeric columns may include more than just StringType
val indexed_columns = categorical_columns.map(_ + "_indexed") //rename the indexed columns
val numeric_columns = (df.dtypes.filter(_._2 != "StringType").map(_._1)).filter(! _.equals("churn")) //churn is label, exclude it from feature columns

val indexers = categorical_columns.map(column =>
  new StringIndexer()
    .setInputCol(column)
    .setOutputCol(column+"_indexed")) 
val assembler = new VectorAssembler()
  .setInputCols(numeric_columns ++ indexed_columns)
  .setOutputCol("features")
val dt = new DecisionTreeClassifier()
  .setLabelCol("churn")
  .setFeaturesCol("features")
  .setMaxBins(50)
  .setSeed(42)
val pipeline = new Pipeline().setStages(indexers :+ assembler :+ dt)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Split data into training and test data sets, train and evaluate the model

// COMMAND ----------

val Array(trainingdf, testdf) = df.randomSplit(Array(0.7, 0.3), seed = 42)

val model = pipeline.fit(trainingdf)
val predictions = model.transform(testdf)

val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("churn")
  .setMetricName("areaUnderROC")
val aur = evaluator.evaluate(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Tune the model with hyperparameter tuning and cross validation

// COMMAND ----------

val paramGrid = new ParamGridBuilder()
  .addGrid(dt.maxDepth, Array(10, 20))
  .build()

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)

val model = cv.fit(trainingdf)
val predictions = model.transform(testdf)
val aur = evaluator.evaluate(predictions)
println("best aur = " + aur)

val bestModel = model.bestModel.asInstanceOf[PipelineModel]
val stages = bestModel.stages
val dtm = stages(stages.length-1).asInstanceOf[DecisionTreeClassificationModel]
println("maxDepth = " + dtm.getMaxDepth)