// Databricks notebook source
// MAGIC %md
// MAGIC This example creates a machine learning model that predicts whether there will be a click for a given ad impression. It uses the [Click-Through Rate Prediction dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data) from Kaggle.  It's a Sacla version of the [Python implementation by Databricks](https://s3.us-east-2.amazonaws.com/databricks-dennylee/notebooks/advertising-analytics-click-prediction.dbc). 

// COMMAND ----------

import org.apache.spark.sql.types._
val dataSchema = StructType(
  StructField("id", DecimalType(20,0)) :: 
  StructField("click", IntegerType) :: 
  StructField("hour", IntegerType) ::
  StructField("C1", IntegerType) ::
  StructField("banner_pos", IntegerType) ::
  StructField("site_id", StringType) ::
  StructField("site_domain", StringType) ::
  StructField("site_category", StringType) ::
  StructField("app_id", StringType) ::
  StructField("app_domain", StringType) ::
  StructField("app_category", StringType) ::
  StructField("device_id", StringType) ::
  StructField("device_ip", StringType) ::
  StructField("device_model", StringType) ::
  StructField("device_type", IntegerType) ::
  StructField("device_conn_type", IntegerType) ::
  StructField("C14", IntegerType) ::
  StructField("C15", IntegerType) ::
  StructField("C16", IntegerType) ::
  StructField("C17", IntegerType) ::
  StructField("C18", IntegerType) ::
  StructField("C19", IntegerType) ::
  StructField("C20", IntegerType) ::
  StructField("C21", IntegerType) ::
  Nil
  )

// read from gz csv files on mounted ADLS
val df = spark.read.
  schema(dataSchema).
  option("header", true).
  csv("/mnt/mldata/kaggle/train.gz")

display(df)

// COMMAND ----------

// convert to parquet files for better performance if we need to run this again
df.write.mode("overwrite").parquet("/mnt/mldata/kaggle/trainpq")

// COMMAND ----------

import org.apache.spark.sql.functions._

val rawImpression = spark.read.parquet("/mnt/mldata/kaggle/trainpq")
val impression = rawImpression.withColumn("hr", substring($"hour", 7, 2).cast(IntegerType))

display(impression)

// COMMAND ----------

val bp = impression.groupBy($"banner_pos").agg(sum("click") / (count("banner_pos") * 1.0) as "ctr").orderBy($"banner_pos")
display(bp)

// COMMAND ----------

val hr = impression.groupBy($"hr").agg(sum("click")/count("hour")* 1.0 as "ctr").orderBy($"hr")
display(hr)

// COMMAND ----------

//impression.dtypes is an array of (columnName, columnType) 
//impression.dtypes.foreach(println)

val strCols = impression.dtypes.filter(_._2 == "StringType").map(_._1)
val intCols = impression.dtypes.filter(_._2 == "IntegerType").map(_._1)
strCols.foreach(println)
intCols.foreach(println)

// COMMAND ----------

// these are arrays of (columnname, distinct column values)
val strColsDistinctCount = strCols.map(c => (c, impression.select(countDistinct(c)).collect()(0)(0).toString.toInt)).sortBy(_._2)
strColsDistinctCount.foreach(println)
val intColsDistinctCount = intCols.map(c => (c, impression.select(countDistinct(c)).collect()(0)(0).toString.toInt)).sortBy(_._2)
intColsDistinctCount.foreach(println)


// COMMAND ----------

// remove all the columns that have more than 70 unique values
// remove label column
val maxBins = 70
var categorical = (strColsDistinctCount.filter(_._2 <= maxBins).map(_._1) ++ intColsDistinctCount.filter(_._2 <= maxBins).map(_._1)).
  filter(_ != "click")

// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val stringIndexers = categorical.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_idx"))
val assemblerInput = categorical.map(c => c + "_idx")
val vectorAssembler = new VectorAssembler().setInputCols(assemblerInput).setOutputCol("features")
val labelStringIndexer = new StringIndexer().setInputCol("click").setOutputCol("label")
val pipeline = new Pipeline().setStages(stringIndexers ++ Array(vectorAssembler, labelStringIndexer))
val featurizer = pipeline.fit(impression)
val featurizedImpressions = featurizer.transform(impression)
display(featurizedImpressions.select($"features", $"label"))

// COMMAND ----------

val Array(training, test) = featurizedImpressions.select($"features", $"label").randomSplit(Array(0.7, 0.3), seed = 42)
val gbt = new GBTClassifier().
  setLabelCol("label").
  setFeaturesCol("features").
  setMaxBins(maxBins).
  setMaxDepth(10).
  setMaxIter(10).
  setSeed(42)
val model = gbt.fit(training)

// COMMAND ----------

model.write.overwrite().save("/mnt/mldata/kaggle/gbtmodel42")

// COMMAND ----------

import org.apache.spark.ml.classification.GBTClassificationModel

val model = GBTClassificationModel.load("/mnt/mldata/kaggle/gbtmodel")

// COMMAND ----------

val predictions = model.transform(test)
val ev = new BinaryClassificationEvaluator().
  setRawPredictionCol("rawPrediction").
  setMetricName("areaUnderROC")
ev.evaluate(predictions)

// COMMAND ----------

val accuracy = predictions.agg(sum(when($"prediction" === $"label", 1).otherwise(0)) / (count("label") * 1.0) as "accuracy")
display(accuracy)

// COMMAND ----------

import scala.util.parsing.json._

val featuremeta =  predictions.schema("features").metadata.
  getMetadata("ml_attr").
  getMetadata("attrs").
  json
val featuredf = spark.read.json(Seq(featuremeta).toDS).
  withColumn("features", explode($"nominal")).
  select($"features.idx", $"features.name")
val weights = model.featureImportances.toArray.zipWithIndex
val weightdf = spark.createDataset(weights).toDF.
  select($"_2" as "idx", $"_1" as "weight")
display(featurenamedf.join(weightdf, featurenamedf("idx") === weightdf("idx")).orderBy(desc("weight")))
