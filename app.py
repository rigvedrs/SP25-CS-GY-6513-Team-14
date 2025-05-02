import streamlit as st
import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from datetime import datetime
import os

st.set_page_config(page_title="Purchase Prediction App", layout="wide")
st.title("Customer Purchase Prediction")

# File upload widgets
model_path = st.text_input("Path to model directory:", "./RF_model")
csv_path = st.text_input("Path to CSV file:", "./data.csv")

if st.button("Run Prediction"):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Initializing Spark Session...")
    progress_bar.progress(10)
    
    # Create SparkSession
    try:
        spark = SparkSession.builder \
            .appName("StreamlitModelPrediction") \
            .config("spark.sql.repl.eagerEval.enabled", True) \
            .getOrCreate()
        
        # Define UDFs for preprocessing
        @udf(returnType=IntegerType())
        def is_purchased_label(purchase):
            if purchase == "purchase":
                return 1
            return 0

        @udf(returnType=IntegerType())
        def week(s):
            return datetime.strptime(str(s)[0:10], "%Y-%m-%d").weekday()

        @udf
        def extract_category(category, brand):
            newlist = str(category).split('.')
            if newlist[0] == "empty":
                if brand == "empty":
                    return "unknown"
                return brand
            return newlist[0]

        @udf
        def extract_product(category, brand):
            newlist = str(category).split('.')
            if newlist[-1] == "empty":
                if brand == "empty":
                    return "unknown"
                return brand
            return newlist[-1]
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"Model path does not exist: {model_path}")
            st.stop()
        
        if not os.path.exists(csv_path):
            st.error(f"CSV file does not exist: {csv_path}")
            st.stop()
        
        status_text.text("Loading model...")
        progress_bar.progress(20)
        
        # Load the model
        model = RandomForestClassificationModel.load(model_path)
        
        status_text.text("Loading CSV data...")
        progress_bar.progress(30)
        
        # Load the CSV data
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .format("csv") \
            .load(csv_path)
        
        status_text.text("Preprocessing data...")
        progress_bar.progress(40)
        
        # Preprocessing steps
        # 1. Fill missing values
        df = df.na.fill(value="empty", subset=["category_code", "brand"])

        # 2. Extract category and product from category_code
        df = df.select("*", extract_category("category_code", "brand").alias("category"), 
                    extract_product("category_code", "brand").alias("product"))
        df = df.drop("category_code")

        # 3. Extract time features
        df = df.withColumn('Date', split(df['event_time'], ' ').getItem(0)) \
            .withColumn('Time', split(df['event_time'], ' ').getItem(1))
        df = df.withColumn('Day', split(df['Date'], '-').getItem(2)) \
            .withColumn('Hour', split(df['Time'], ':').getItem(0))
        df = df.drop("Date")

        status_text.text("Filtering events...")
        progress_bar.progress(50)
        
        # 4. Filter for cart and purchase events
        cart_purchase_df = df.filter("event_type == 'cart' OR event_type == 'purchase'")
        distinct_cart_purchase = cart_purchase_df.drop_duplicates(subset=['event_type', 'product_id', 'user_id', 'user_session'])

        # 5. Create session activity count
        activity_in_session = cart_purchase_df.groupby(['user_session']).count()

        # 6. Create target variable and join with activity count
        df_targets = distinct_cart_purchase.select("*", is_purchased_label("event_type").alias("is_purchased"))
        df_targets = df_targets.join(activity_in_session, on="user_session", how="left")

        # 7. Add week feature
        df_targets_week = df_targets.select("*", week("event_time").alias("week"))
        df_targets_week = df_targets_week.dropDuplicates(["user_session"])

        # 8. Select the same features used in training
        features = df_targets_week.select("user_id", "event_type", "brand", "price", "count", "week", "category", "product", "is_purchased")
        features = features.na.drop()
        
        status_text.text("Building transformation pipeline...")
        progress_bar.progress(60)
        
        # Create the transformation pipeline
        # StringIndexers
        categotyIdxer = StringIndexer(inputCol='category', outputCol='category_idx')
        event_typeIdxer = StringIndexer(inputCol='event_type', outputCol='event_type_idx')
        brandIdxer = StringIndexer(inputCol='brand', outputCol='brand_idx')
        productIdxer = StringIndexer(inputCol='product', outputCol='product_idx')
        labelIndexer = StringIndexer(inputCol="is_purchased", outputCol="label")

        # OneHotEncoders
        one_hot_encoder_category = OneHotEncoder(inputCol="category_idx", outputCol="category_vec")
        one_hot_encoder_product = OneHotEncoder(inputCol="product_idx", outputCol="product_vec")
        one_hot_encoder_brand = OneHotEncoder(inputCol="brand_idx", outputCol="brand_vec")
        one_hot_encoder_event_type = OneHotEncoder(inputCol="event_type_idx", outputCol="event_type_vec")

        # Indexer stages
        stages_indexer = [
            categotyIdxer,
            event_typeIdxer,
            brandIdxer,
            productIdxer,
            labelIndexer
        ]

        # One-hot encoder stages
        stages_one_hot = [
            one_hot_encoder_category,
            one_hot_encoder_event_type,
            one_hot_encoder_brand,
            one_hot_encoder_product
        ]

        # Vector assemblers
        assembler_cat = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in stages_one_hot], 
                                        outputCol="features_cat")
        num_cols = ["count", "week", "price"]
        assemblerNum = VectorAssembler(inputCols=num_cols, outputCol="features_num")
        final_assembler = VectorAssembler(inputCols=["features_cat", "features_num"], outputCol="features")

        # Create pipeline
        pipeline = Pipeline(stages=stages_indexer + stages_one_hot + [assembler_cat] + [assemblerNum] + [final_assembler])
        
        status_text.text("Applying transformations...")
        progress_bar.progress(70)
        
        # Apply transformations
        df_transformed = pipeline.fit(features).transform(features)
        
        status_text.text("Making predictions...")
        progress_bar.progress(80)
        
        # Make predictions - only select features and user_id (not label)
        predictions = model.transform(df_transformed.select("features", "user_id"))
        
        # Format predictions for display (without showing actual values)
        predictions_by_user = predictions.select(
            "user_id", 
            when(col("prediction") == 1, "Will Purchase").otherwise("Won't Purchase").alias("prediction")
        )
        
        # Group by user_id to get one prediction per user
        window_spec = Window.partitionBy("user_id").orderBy("user_id")
        user_predictions = predictions_by_user.withColumn("row_number", row_number().over(window_spec)) \
                                             .filter(col("row_number") == 1) \
                                             .drop("row_number")
        
        status_text.text("Preparing results...")
        progress_bar.progress(90)
        
        # Convert to pandas for display
        pandas_df = user_predictions.orderBy("user_id").toPandas()
        
        # Summary statistics
        users_count = len(pandas_df)
        purchase_predictions_count = len(pandas_df[pandas_df["prediction"] == "Will Purchase"])
        purchase_percentage = (purchase_predictions_count/users_count)*100 if users_count > 0 else 0
        
        # Display results
        st.subheader("Prediction Results")
        st.dataframe(pandas_df)
        
        # Display summary
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", users_count)
        col2.metric("Purchase Predictions", purchase_predictions_count)
        col3.metric("Purchase Percentage", f"{purchase_percentage:.2f}%")
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Prediction complete!")
        
    except Exception as e:
        import traceback
        st.error(f"Error during prediction process: {str(e)}")
        st.error(traceback.format_exc())

st.markdown("""
### Instructions
1. Enter the path to your model directory
2. Enter the path to your CSV file
3. Click 'Run Prediction' to process the data and see results
""")