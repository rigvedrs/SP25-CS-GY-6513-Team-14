import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BooleanType
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import TimestampType
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, lit, dayofmonth, hour, count, monotonically_increasing_id
from pyspark.sql.types import IntegerType
from pyspark.sql.types import BooleanType
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StructType, StructField
from datetime import datetime
import pandas as pd
import sys
import os

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, element_at, split, dayofmonth, hour, count, monotonically_increasing_id, when
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BooleanType
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import udf
from datetime import datetime
import pandas as pd
import sys
import os

# ASCII art header
print("""
╔════════════════════════════════════════════════════════════╗
║         🛒  E-COMMERCE PURCHASE PREDICTOR  🛒              ║
║                                                            ║
║  Predict whether cart items will be purchased              ║
╚════════════════════════════════════════════════════════════╝
""")

# Input and model paths
input_csv = "user_history_for_prediction.csv"
model_path = "./RF_model"

if not os.path.exists(input_csv):
    print(f"Error: File '{input_csv}' not found.")
    sys.exit(1)

if not os.path.exists(model_path):
    print(f"Error: Model path '{model_path}' not found.")
    sys.exit(1)

print(f"✓ Input file: {input_csv}")
print(f"✓ Model path: {model_path}")

# Create SparkSession with minimal output
spark = SparkSession.builder \
    .appName("SimplePrediction") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

# Suppress INFO logs
spark.sparkContext.setLogLevel("ERROR")

# Load data
print("Loading data...")
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .format("csv") \
    .load(input_csv)

print(f"Loaded {df.count()} rows")

# Filter for cart items only
cart_df = df.filter(col("event_type") == "cart")
print(f"Found {cart_df.count()} cart items for prediction")

# Display sample
print("\nSample data:")
cart_df.select("event_type", "category", "product", "brand", "price").show(3)

# Prepare features for model
print("Preparing features...")

# Make sure required columns exist
required_columns = ["event_type", "brand", "price", "count", "week", "category", "product"]
for column in required_columns:
    if column not in cart_df.columns:
        if column in ["price", "count", "week"]:
            cart_df = cart_df.withColumn(column, lit(0.0))
        else:
            cart_df = cart_df.withColumn(column, lit("unknown"))

# Check column diversity
for column in ["category", "brand", "product"]:
    distinct_count = cart_df.select(column).distinct().count()
    print(f"Column {column} has {distinct_count} unique values")
    if distinct_count < 2:
        print(f"  Warning: Column {column} needs at least 2 unique values")
        # Add a dummy row with a different value if needed
        dummy_value = f"dummy_{column}"
        dummy_row = cart_df.limit(1).withColumn(column, lit(dummy_value))
        cart_df = cart_df.union(dummy_row)
        print(f"  Added dummy row with {column}='{dummy_value}'")

# Add is_purchased column (will be 0 for all cart items)
cart_df = cart_df.withColumn("is_purchased", lit(0))

# Create a temporary view for manual SQL approach
cart_df.createOrReplaceTempView("cart_items")

# Use SQL to shape the data - avoids complex transformations
features_df = spark.sql("""
    SELECT 
        event_type, 
        brand, 
        price, 
        COALESCE(count, 1) as count, 
        COALESCE(week, 0) as week, 
        category, 
        product, 
        is_purchased,
        product_id,
        user_id,
        user_session,
        COALESCE(hour, 0) as hour,
        COALESCE(day, 1) as day
    FROM cart_items
""")

# Load the model
print("Loading model...")
try:
    # Attempt to load the model directly
    model = RandomForestClassificationModel.load(model_path)
    print("Successfully loaded RandomForest model")
except Exception as e:
    print(f"Error loading model directly: {e}")
    try:
        # Try loading as a full pipeline model
        model = PipelineModel.load(model_path)
        print("Successfully loaded Pipeline model")
    except Exception as e2:
        print(f"Error loading as Pipeline model: {e2}")
        print("Using manual feature engineering instead")
        model = None

# Define feature engineering pipeline
feature_stages = []

# A. String indexers for categorical columns
indexers = {}
for col_name in ["category", "brand", "product", "event_type"]:
    output_col = f"{col_name}_idx"
    indexer = StringIndexer(inputCol=col_name, outputCol=output_col, handleInvalid="keep")
    feature_stages.append(indexer)
    
    # Add encoder
    encoder_output = f"{col_name}_vec"
    encoder = OneHotEncoder(inputCol=output_col, outputCol=encoder_output, dropLast=False)
    feature_stages.append(encoder)
    indexers[col_name] = encoder_output

# B. Numeric features
numeric_cols = ["price", "count", "week"]

# C. Create assemblers
assembler_cat = VectorAssembler(
    inputCols=list(indexers.values()), 
    outputCol="features_cat"
)
feature_stages.append(assembler_cat)

assembler_num = VectorAssembler(
    inputCols=numeric_cols, 
    outputCol="features_num"
)
feature_stages.append(assembler_num)

final_assembler = VectorAssembler(
    inputCols=["features_cat", "features_num"], 
    outputCol="features"
)
feature_stages.append(final_assembler)

# Apply feature pipeline
feature_pipeline = Pipeline(stages=feature_stages)
print("Transforming features...")
try:
    transformed_df = feature_pipeline.fit(features_df).transform(features_df)
    
    # If we don't have a model, save features to CSV and exit
    if model is None:
        print("No model available. Saving processed features to CSV.")
        # Convert to pandas and save
        pd_features = transformed_df.toPandas()
        pd_features.to_csv("processed_features.csv", index=False)
        print("Saved processed features to processed_features.csv")
        sys.exit(0)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.transform(transformed_df)
    
    # Extract predictions with original data - ONLY binary prediction
    results = predictions.select(
        col("product_id"),
        col("category"),
        col("product"),
        col("brand"),
        col("price"),
        col("user_id"),
        col("user_session"),
        col("hour"),
        col("day"),
        col("prediction").alias("will_purchase")
    )
    
    # Convert to pandas for nice display
    pd_results = results.toPandas()
    
    # Save to CSV
    output_file = "purchase_predictions.csv"
    pd_results.to_csv(output_file, index=False)
    
    # Display results
    print(f"\n✅ Predictions complete! Saved to {output_file}")
    
    # Format for display
    formatted = pd_results.copy()
    formatted["will_purchase"] = formatted["will_purchase"].map({1.0: "YES ✓", 0.0: "NO ✗"})
    
    # Display sample
    display_cols = ["product", "category", "brand", "price", "will_purchase"]
    print("\n==== SAMPLE PREDICTIONS ====")
    print(formatted[display_cols].head(10))
    
    # Summary stats
    total = len(pd_results)
    likely = len(pd_results[pd_results["will_purchase"] > 0.5])
    print(f"\n📊 SUMMARY: Out of {total} items in cart, {likely} ({likely/total*100:.1f}%) are predicted to be purchased.")
    
except Exception as e:
    print(f"Error during processing: {e}")
    import traceback
    traceback.print_exc()

# Stop Spark
spark.stop()
print("\nDone!")