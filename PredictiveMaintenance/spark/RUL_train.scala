// Databricks notebook source
// MAGIC %md
// MAGIC # Predictive maintenance
// MAGIC This example predicts remaining useful life of aircraft engines from historical sensor data.
// MAGIC 
// MAGIC Download the "__Turbofan Engine Degradation Simulation Data Set__" from [NASA Ames Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val dataSchema = StructType(
  StructField("id", IntegerType) :: 
  StructField("cycle", IntegerType) :: 
  StructField("setting1", FloatType) ::
  StructField("setting2", FloatType) :: 
  StructField("setting3", FloatType) :: 
  StructField("s1", FloatType) :: 
  StructField("s2", FloatType) :: 
  StructField("s3", FloatType) :: 
  StructField("s4", FloatType) :: 
  StructField("s5", FloatType) :: 
  StructField("s6", FloatType) :: 
  StructField("s7", FloatType) :: 
  StructField("s8", FloatType) :: 
  StructField("s9", FloatType) :: 
  StructField("s10", FloatType) :: 
  StructField("s11", FloatType) :: 
  StructField("s12", FloatType) :: 
  StructField("s13", FloatType) :: 
  StructField("s14", FloatType) :: 
  StructField("s15", FloatType) :: 
  StructField("s16", FloatType) :: 
  StructField("s17", FloatType) :: 
  StructField("s18", FloatType) :: 
  StructField("s19", FloatType) :: 
  StructField("s20", FloatType) :: 
  StructField("s21", FloatType) :: 
  Nil
  )

val labelSchema = StructType(
  StructField("RUL", IntegerType) :: 
  Nil)

// COMMAND ----------

// MAGIC %md
// MAGIC ![Alt text](https://github.com/liupeirong/sparksummit2018ml/blob/dev/PredictiveMaintenance/images/1prepTrain.PNG?raw=true)

// COMMAND ----------

val train_raw = spark.read.schema(dataSchema).option("delimiter", " ").csv("/mnt/mldata/train_FD001.txt")
val train_maxcycle = train_raw.groupBy("id").max("cycle").select($"id".alias("id_maxcycle"), $"max(cycle)".alias("maxcycle"))
val train_labeled = train_raw.join(train_maxcycle, $"id" === $"id_maxcycle").withColumn("RUL", $"maxcycle" - $"cycle")
val train_df = train_labeled.select("id", "cycle", "s9", "s11", "s14", "s15", "RUL")
display(train_df)

// COMMAND ----------

// MAGIC %md
// MAGIC ![Alt text](https://github.com/liupeirong/sparksummit2018ml/blob/dev/PredictiveMaintenance/images/2train.PNG?raw=true)

// COMMAND ----------

val assembler = new VectorAssembler()
  .setInputCols(Array("cycle", "s9", "s11", "s14", "s15"))
  .setOutputCol("features")

val gbt = new GBTRegressor()
  .setLabelCol("RUL")
  .setFeaturesCol("features")
  .setMaxIter(100)
  .setMaxDepth(5)
  .setStepSize(0.1)
  .setSeed(5)

val pipeline = new Pipeline().setStages(Array(assembler, gbt))

val model = pipeline.fit(train_df)
model.asInstanceOf[PipelineModel].stages(1).asInstanceOf[GBTRegressionModel].toDebugString

// COMMAND ----------

// MAGIC %md
// MAGIC ![Alt text](https://github.com/liupeirong/sparksummit2018ml/blob/dev/PredictiveMaintenance/images/3prepTest.PNG?raw=true)

// COMMAND ----------

val test_raw = spark.read.schema(dataSchema).option("delimiter", " ").csv("/mnt/mldata/test_FD001.txt")
val test_maxcycle = test_raw.groupBy("id").max("cycle").select($"id".alias("id_maxcycle"), $"max(cycle)".alias("maxcycle"))
val test_maxconly = test_raw.join(test_maxcycle, $"id" === $"id_maxcycle" && $"cycle" === $"maxcycle")
val test_ordered = test_maxconly.orderBy("id").select("id", "cycle", "s9", "s11", "s14", "s15")

val rul_raw = spark.read.schema(labelSchema).option("delimiter", " ").csv("/mnt/mldata/RUL_FD001.txt")
val rul_ordered = rul_raw.withColumn("id_ordered", monotonically_increasing_id()+1)
val test_df = test_ordered.join(rul_ordered, $"id" === $"id_ordered").
  select("id", "cycle", "s9", "s11", "s14", "s15", "RUL")
display(test_df)

// COMMAND ----------

// MAGIC %md
// MAGIC ![Alt text](https://github.com/liupeirong/sparksummit2018ml/blob/dev/PredictiveMaintenance/images/4evaluate.PNG?raw=true)

// COMMAND ----------

val predictions = model.transform(test_df)

val evaluator = new RegressionEvaluator()
  .setLabelCol("RUL")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
rmse

// COMMAND ----------

// MAGIC %md
// MAGIC ![Alt text](https://github.com/liupeirong/sparksummit2018ml/blob/dev/PredictiveMaintenance/images/5tune.PNG?raw=true)

// COMMAND ----------

val paramGrid = new ParamGridBuilder()
  .addGrid(gbt.maxIter, Array(10, 50, 100))
  .addGrid(gbt.maxDepth, Array(5, 10))
  .addGrid(gbt.stepSize, Array(0.1, 0.2))
  .build()

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setParallelism(9)

val model = cv.fit(train_df)
val predictions = model.transform(test_df)
val rmse = evaluator.evaluate(predictions)

val bestModel = model.bestModel.asInstanceOf[PipelineModel]
val stages = bestModel.stages
val gbtModel = stages(1).asInstanceOf[GBTRegressionModel]
println("best model: maxDepth = " + gbtModel.getMaxDepth + 
        ", maxIter = " + gbtModel.getMaxIter +
        ", stepSize = " + gbtModel.getStepSize +
        ", rmse = " + rmse)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Save the model

// COMMAND ----------

bestModel.write.overwrite().save("/mnt/mldata/rul_gbt_regression_model")