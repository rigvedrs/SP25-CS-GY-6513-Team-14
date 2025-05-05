import streamlit as st
import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, countDistinct, split, when, row_number, count, desc, rank
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback
import builtins
import circlify
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="E-commerce Analytics & Prediction", layout="wide")
st.title("E-commerce Analytics and Customer Purchase Prediction")

# Initialize SparkSession
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("StreamlitEcommerceAnalytics") \
        .config("spark.sql.repl.eagerEval.enabled", True) \
        .getOrCreate()

spark = init_spark()

# Define UDFs for preprocessing
@udf(returnType=IntegerType())
def is_purchased_label(purchase):
    return 1 if purchase == "purchase" else 0

@udf(returnType=IntegerType())
def week(s):
    return datetime.strptime(str(s)[0:10], "%Y-%m-%d").weekday()

@udf
def extract_category(category, brand):
    newlist = str(category).split('.')
    return brand if newlist[0] == "empty" and brand != "empty" else "unknown" if newlist[0] == "empty" else newlist[0]

@udf
def extract_product(category, brand):
    newlist = str(category).split('.')
    return brand if newlist[-1] == "empty" and brand != "empty" else "unknown" if newlist[-1] == "empty" else newlist[-1]

@udf(returnType=FloatType())
def cart_miss_rate_udf(cart_count, purchase_count):
    if cart_count is None or cart_count == 0 or purchase_count is None:
        return 0.0
    return (1 - purchase_count / cart_count) * 100

# Tabs for Predictions and Analytics
tab1, tab2 = st.tabs(["Predictions", "Analytics"])

# Predictions Tab
with tab1:
    st.header("Customer Purchase Prediction")
    col1, col2 = st.columns(2)
    with col1:
        model_path = st.text_input("Path to model directory:", "./RF_model")
    with col2:
        csv_path = st.text_input("Path to CSV file:", "./data_without_purchase_event.csv")

    if st.button("Run Prediction"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Validate file paths
            if not os.path.exists(model_path):
                st.error(f"Model path does not exist: {model_path}")
                st.stop()
            if not os.path.exists(csv_path):
                st.error(f"CSV file does not exist: {csv_path}")
                st.stop()

            status_text.text("Loading model...")
            progress_bar.progress(20)
            model = RandomForestClassificationModel.load(model_path)

            status_text.text("Loading CSV data...")
            progress_bar.progress(30)
            df = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load(csv_path)

            status_text.text("Preprocessing data...")
            progress_bar.progress(40)
            # Preprocessing
            df = df.na.fill(value="empty", subset=["category_code", "brand"])
            df = df.select("*", extract_category("category_code", "brand").alias("category"),
                           extract_product("category_code", "brand").alias("product")).drop("category_code")
            df = df.withColumn('Date', split(df['event_time'], ' ').getItem(0)) \
                   .withColumn('Time', split(df['event_time'], ' ').getItem(1))
            df = df.withColumn('Day', split(df['Date'], '-').getItem(2)) \
                   .withColumn('Hour', split(df['Time'], ':').getItem(0)).drop("Date")

            status_text.text("Filtering events...")
            progress_bar.progress(50)
            cart_purchase_df = df.filter(col("event_type").isin(["cart", "purchase"]))
            distinct_cart_purchase = cart_purchase_df.drop_duplicates(subset=['event_type', 'product_id', 'user_id', 'user_session'])
            activity_in_session = cart_purchase_df.groupby(['user_session']).count()

            df_targets = distinct_cart_purchase.select("*", is_purchased_label("event_type").alias("is_purchased"))
            df_targets = df_targets.join(activity_in_session, on="user_session", how="left")
            df_targets_week = df_targets.select("*", week("event_time").alias("week")).dropDuplicates(["user_session"])

            features = df_targets_week.select("user_id", "event_type", "brand", "price", "count", "week", "category", "product", "is_purchased")
            features = features.na.drop()

            status_text.text("Building transformation pipeline...")
            progress_bar.progress(60)
            # Transformation pipeline
            categotyIdxer = StringIndexer(inputCol='category', outputCol='category_idx')
            event_typeIdxer = StringIndexer(inputCol='event_type', outputCol='event_type_idx')
            brandIdxer = StringIndexer(inputCol='brand', outputCol='brand_idx')
            productIdxer = StringIndexer(inputCol='product', outputCol='product_idx')
            labelIndexer = StringIndexer(inputCol="is_purchased", outputCol="label")

            one_hot_encoder_category = OneHotEncoder(inputCol="category_idx", outputCol="category_vec")
            one_hot_encoder_product = OneHotEncoder(inputCol="product_idx", outputCol="product_vec")
            one_hot_encoder_brand = OneHotEncoder(inputCol="brand_idx", outputCol="brand_vec")
            one_hot_encoder_event_type = OneHotEncoder(inputCol="event_type_idx", outputCol="event_type_vec")

            stages_indexer = [categotyIdxer, event_typeIdxer, brandIdxer, productIdxer, labelIndexer]
            stages_one_hot = [one_hot_encoder_category, one_hot_encoder_event_type, one_hot_encoder_brand, one_hot_encoder_product]

            assembler_cat = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in stages_one_hot], outputCol="features_cat")
            num_cols = ["count", "week", "price"]
            assemblerNum = VectorAssembler(inputCols=num_cols, outputCol="features_num")
            final_assembler = VectorAssembler(inputCols=["features_cat", "features_num"], outputCol="features")

            pipeline = Pipeline(stages=stages_indexer + stages_one_hot + [assembler_cat, assemblerNum, final_assembler])

            status_text.text("Applying transformations...")
            progress_bar.progress(70)
            df_transformed = pipeline.fit(features).transform(features)

            status_text.text("Making predictions...")
            progress_bar.progress(80)
            predictions = model.transform(df_transformed.select("features", "user_id"))
            predictions_by_user = predictions.select(
                "user_id",
                when(col("prediction") == 1, "Will Purchase").otherwise("Won't Purchase").alias("prediction")
            )

            window_spec = Window.partitionBy("user_id").orderBy("user_id")
            user_predictions = predictions_by_user.withColumn("row_number", row_number().over(window_spec)) \
                                                 .filter(col("row_number") == 1).drop("row_number")

            status_text.text("Preparing results...")
            progress_bar.progress(90)
            pandas_df = user_predictions.orderBy("user_id").toPandas()

            users_count = len(pandas_df)
            purchase_predictions_count = len(pandas_df[pandas_df["prediction"] == "Will Purchase"])
            purchase_percentage = (purchase_predictions_count / users_count) * 100 if users_count > 0 else 0

            st.subheader("Prediction Results")
            st.dataframe(pandas_df)

            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Users", users_count)
            col2.metric("Purchase Predictions", purchase_predictions_count)
            col3.metric("Purchase Percentage", f"{purchase_percentage:.2f}%")

            progress_bar.progress(100)
            status_text.text("Prediction complete!")

        except Exception as e:
            st.error(f"Error during prediction process: {str(e)}")
            st.error(traceback.format_exc())

    st.markdown("""
    ### Instructions
    1. Enter the path to your model directory
    2. Enter the path to your CSV file
    3. Click 'Run Prediction' to process the data and see results
    """)

with tab2:
    st.header("📊 E-commerce Analytics Dashboard")
    csv_path_analytics = "./data_without_purchase_event.csv"

    if st.button("Run Analytics"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            if not os.path.exists(csv_path_analytics):
                st.error(f"CSV file does not exist: {csv_path_analytics}")
                st.stop()

            status_text.text("Loading CSV data...")
            progress_bar.progress(20)
            df = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load(csv_path_analytics)

            status_text.text("Preprocessing data...")
            progress_bar.progress(30)
            df = df.na.fill(value="empty", subset=["category_code", "brand"])
            df = df.select("*", extract_category("category_code", "brand").alias("category"),
                           extract_product("category_code", "brand").alias("product")).drop("category_code")
            df = df.withColumn('Date', split(df['event_time'], ' ').getItem(0)) \
                   .withColumn('Time', split(df['event_time'], ' ').getItem(1))
            df = df.withColumn('Day', split(df['Date'], '-').getItem(2)) \
                   .withColumn('Hour', split(df['Time'], ':').getItem(0)).drop("Date")

            df_view = df.filter(col("event_type") == "view")
            df_cart = df.filter(col("event_type") == "cart")
            df_purchase = df.filter(col("event_type") == "purchase")

            # Unique Visitors
            status_text.text("Calculating unique visitors...")
            progress_bar.progress(40)
            unique_visitors = df.select(countDistinct("user_id")).collect()[0][0]
            st.subheader("👥 Unique Visitors")
            st.metric("Total Unique Visitors in October", unique_visitors)

            # Layout for plots
            col1, col2 = st.columns(2)

            # Funnel Analysis
            with col1:
                status_text.text("Generating funnel analysis...")
                progress_bar.progress(50)
                funnel_data = pd.DataFrame({
                    'event_type': ["View", "Cart", "Purchase"],
                    'count': [df_view.count(), df_cart.count(), df_purchase.count()]
                })
                fig_funnel = go.Figure(go.Funnel(
                    y=funnel_data['event_type'],
                    x=funnel_data['count'],
                    textinfo="value+percent initial",
                    marker=dict(color=["#FFD700", "#FF6347", "#32CD32"]),
                    connector={"line": {"color": "#444", "width": 2}},
                    hoverinfo="x+y+percent initial",
                    textfont=dict(color="#000000")  # Set funnel text to dark black
                ))
                fig_funnel.update_layout(
                    title="User Behavior Funnel",
                    title_x=0.5,
                    title_font=dict(color="#000000"),  # Set title text to dark black
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(tickfont=dict(color="#000000")),  # Set x-axis tick labels to dark black
                    yaxis=dict(tickfont=dict(color="#000000"))   # Set y-axis tick labels to dark black
                )
                st.plotly_chart(fig_funnel, use_container_width=True)

            # Top 10 Categories Browsed
            with col2:
                status_text.text("Analyzing top categories browsed...")
                progress_bar.progress(60)
                df_cat_browsed = df.filter(col("event_type").isin(["view", "cart"]) & (col("category") != "unknown"))
                df_cat_browsed_count = df_cat_browsed.groupBy("category").count().orderBy(desc("count")).limit(10)
                browsed_data = df_cat_browsed_count.toPandas()
                fig_browsed = px.bar(
                    browsed_data,
                    x='category',
                    y='count',
                    text='count',
                    color_discrete_sequence=['#4682B4']
                )
                fig_browsed.update_traces(
                    textposition='outside',
                    marker=dict(line=dict(color='#000', width=1)),
                    textfont=dict(color="#000000")  # Set bar text to dark black
                )
                fig_browsed.update_layout(
                    title="Top 10 Categories Browsed",
                    title_font=dict(color="#000000"),  # Set title text to dark black
                    xaxis_title="Category",
                    yaxis_title="Browse Count",
                    xaxis_tickangle=45,
                    xaxis=dict(tickfont=dict(color="#000000")),  # Set x-axis tick labels to dark black
                    yaxis=dict(tickfont=dict(color="#000000")),  # Set y-axis tick labels to dark black
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_browsed, use_container_width=True)

            # Top 10 Categories Purchased
            with col1:
                status_text.text("Analyzing top categories that may be purchased...")
                progress_bar.progress(70)
                df_cat_purchased = df_purchase.filter(col("category") != "unknown")
                df_cat_purchased_count = df_cat_purchased.groupBy("category").count().orderBy(desc("count")).limit(10)
                purchased_data = df_cat_purchased_count.toPandas()
                fig_purchased = px.bar(
                    purchased_data,
                    x='category',
                    y='count',
                    text='count',
                    color_discrete_sequence=['#228B22']
                )
                fig_purchased.update_traces(
                    textposition='outside',
                    marker=dict(line=dict(color='#000', width=1)),
                    textfont=dict(color="#000000")  # Set bar text to dark black
                )
                fig_purchased.update_layout(
                    title="Top 10 Categories Purchased",
                    title_font=dict(color="#000000"),  # Set title text to dark black
                    xaxis_title="Category",
                    yaxis_title="Purchase Count",
                    xaxis_tickangle=45,
                    xaxis=dict(tickfont=dict(color="#000000")),  # Set x-axis tick labels to dark black
                    yaxis=dict(tickfont=dict(color="#000000")),  # Set y-axis tick labels to dark black
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_purchased, use_container_width=True)

            # Circle Packing
            with col2:
                st.markdown("#### Top Brands in Top Categories")
                top_categories = df_cat_purchased_count.select("category", col("count").alias("category_count")).limit(3)
                df_brand_purchased = df_purchase.filter(col("category") != "unknown").groupBy("category", "brand").count()
                window = Window.partitionBy("category").orderBy(desc("count"))
                top_brands = df_brand_purchased.select("*", rank().over(window).alias("rank")) \
                                                .filter(col("rank") <= 3) \
                                                .select("category", "brand", col("count").alias("brand_count"))
                top_brands_data = top_brands.join(top_categories.select("category"), "category", "inner").toPandas()
                top_categories_pandas = top_categories.toPandas()

                if not top_brands_data.empty and not top_categories_pandas.empty:
                    data = [{
                        'id': 'E-commerce',
                        'datum': top_categories_pandas['category_count'].sum(),
                        'children': [
                            {
                                'id': category,
                                'datum': row['category_count'],
                                'children': [
                                    {'id': brand_row['brand'], 'datum': brand_row['brand_count']}
                                    for _, brand_row in top_brands_data[top_brands_data['category'] == category][['brand', 'brand_count']].iterrows()
                                ]
                            }
                            for _, row in top_categories_pandas.iterrows()
                            for category in [row['category']]
                        ]
                    }]

                    circles = circlify.circlify(data, show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1))
                    fig = go.Figure()

                    for circle in circles:
                        if circle.level == 2:  # Categories
                            x, y, r = circle
                            fig.add_shape(type="circle", xref="x", yref="y", x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                                            fillcolor="lightblue", opacity=0.5, line=dict(width=2))
                            fig.add_annotation(x=x, y=y, text=circle.ex["id"], showarrow=False,
                                                font=dict(size=12, color="#000000"),  # Set annotation text to dark black
                                                bgcolor="white", bordercolor="black", borderpad=4)
                        elif circle.level == 3:  # Brands
                            x, y, r = circle
                            fig.add_shape(type="circle", xref="x", yref="y", x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                                            fillcolor="#69b3a2", opacity=0.7, line=dict(width=2))
                            fig.add_annotation(x=x, y=y, text=circle.ex["id"], showarrow=False,
                                                font=dict(size=10, color="#000000"))  # Set annotation text to dark black

                    lim = max(max(abs(circle.x) + circle.r, abs(circle.y) + circle.r) for circle in circles)
                    fig.update_layout(
                        title="Top Brands in Top Categories",
                        title_x=0.5,
                        title_font=dict(color="#000000"),  # Set title text to dark black
                        showlegend=False,
                        xaxis=dict(visible=False, range=[-lim, lim], tickfont=dict(color="#000000")),  # Set x-axis tick labels to dark black
                        yaxis=dict(visible=False, range=[-lim, lim], tickfont=dict(color="#000000")),  # Set y-axis tick labels to dark black
                        height=450,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("**Note**: Larger circles indicate higher purchase counts for brands within top categories.")
                else:
                    st.warning("Insufficient data to generate circle packing visualization.")

            # Purchase Trends
            status_text.text("Analyzing purchase trends...")
            progress_bar.progress(90)
            df_purchase_date_count = df_purchase.groupBy("Day").count().orderBy("Day")
            date_data = df_purchase_date_count.toPandas()
            fig_trends = px.bar(
                data_frame=date_data,
                x='Day',
                y='count',
                text='count',
                color_discrete_sequence=['#9932CC']
            )
            fig_trends.update_traces(
                textposition='outside',
                marker=dict(line=dict(color='#000', width=1)),
                textfont=dict(color="#000000")  # Set bar text to dark black
            )
            fig_trends.update_layout(
                title="Purchase Trends Across the Month",
                title_font=dict(color="#000000"),  # Set title text to dark black
                xaxis_title="Day of Month",
                yaxis_title="Purchase Count",
                xaxis=dict(tickfont=dict(color="#000000")),  # Set x-axis tick labels to dark black
                yaxis=dict(tickfont=dict(color="#000000")),  # Set y-axis tick labels to dark black
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_trends, use_container_width=True)
            st.write("**Analysis**: Purchase interest peaks around mid-month (days 11-16). Consider mid-month sales to boost conversions.")

            # E-commerce Prime Time
            status_text.text("Analyzing prime time...")
            progress_bar.progress(95)
            df_view_hour = df_view.groupBy("Hour").count().withColumnRenamed("count", "view_count")
            df_cart_hour = df_cart.groupBy("Hour").count().withColumnRenamed("count", "cart_count")
            df_purchase_hour = df_purchase.groupBy("Hour").count().withColumnRenamed("count", "purchase_count")
            df_combined_hour = df_view_hour.join(df_cart_hour, "Hour", "left").join(df_purchase_hour, "Hour", "left").na.fill(0).orderBy("Hour")
            hour_data = df_combined_hour.toPandas()
            fig_prime = go.Figure()
            fig_prime.add_trace(go.Bar(
                x=hour_data['Hour'], y=hour_data['view_count'], name='View',
                marker_color='#FFD700', text=hour_data['view_count'], textposition='none',
                textfont=dict(color="#000000")  # Set bar text to dark black
            ))
            fig_prime.add_trace(go.Bar(
                x=hour_data['Hour'], y=hour_data['cart_count'], name='Cart',
                marker_color='#FF6347', text=hour_data['cart_count'], textposition='none',
                textfont=dict(color="#000000")  # Set bar text to dark black
            ))
            fig_prime.add_trace(go.Bar(
                x=hour_data['Hour'], y=hour_data['purchase_count'], name='Purchase',
                marker_color='#32CD32', text=hour_data['purchase_count'], textposition='none',
                textfont=dict(color="#000000")  # Set bar text to dark black
            ))
            fig_prime.update_layout(
                title="E-commerce Prime Time",
                title_font=dict(color="#000000"),  # Set title text to dark black
                xaxis_title="Hour of Day",
                yaxis_title="Count",
                xaxis=dict(tickfont=dict(color="#000000")),  # Set x-axis tick labels to dark black
                yaxis=dict(tickfont=dict(color="#000000")),  # Set y-axis tick labels to dark black
                barmode='stack',
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color="#000000"))  # Set legend text to dark black
            )
            st.plotly_chart(fig_prime, use_container_width=True)
            st.write("**Analysis**: Peak activity occurs around 16:00. A flash sale from 13:00 to 16:00 could increase impulse purchases.")

            progress_bar.progress(100)
            status_text.text("Analytics complete!")

        except Exception as e:
            st.error(f"Error during analytics process: {str(e)}")
            st.error(traceback.format_exc())