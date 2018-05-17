// Databricks notebook source
// MAGIC %md
// MAGIC ## Predict remaining useful life of aircraft engines as data streams in

// COMMAND ----------

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.streaming.Trigger
import scala.concurrent.duration._

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


// COMMAND ----------

// MAGIC %md
// MAGIC ## Load the trained model and make predictions on each row of the input dataframe

// COMMAND ----------

val gbtModel = PipelineModel.load("/mnt/mldata/rul_gbt_regression_model")

val df = spark.readStream.
  option("delimiter", " ").
  schema(dataSchema).
  option("maxFilesPerTrigger", 1).
  csv("/mnt/mldata/test")

val predictions = gbtModel.transform(df)

val query = predictions.select("id", "cycle", "prediction").
  writeStream.
  format("memory").
  queryName("predictedRUL").
  start

// COMMAND ----------

// MAGIC %md
// MAGIC ###An example showing predicted cycle vs. existing input cycle on engine 3 

// COMMAND ----------

// MAGIC %sql select id, cycle, prediction from predictedRUL where id = 3 order by cycle 