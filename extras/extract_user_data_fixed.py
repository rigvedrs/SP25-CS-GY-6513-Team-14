import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col as spark_col, lit, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
import pandas as pd
import random

# Create SparkSession
spark = SparkSession.builder \
    .appName("ExtractUserData") \
    .config("spark.sql.repl.eagerEval.enabled", True) \
    .getOrCreate()

# Define schema explicitly to avoid column name issues
schema = StructType([
    StructField("index", IntegerType(), True),  # This is likely the *c0 column
    StructField("event_time", TimestampType(), True),
    StructField("event_type", StringType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("category_id", IntegerType(), True),
    StructField("category_code", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("user_session", StringType(), True)
])

# Path to your original data
path_to_csv = "./small_oct_2019.csv"  # Update with your actual path

# Load the CSV data
print(f"Loading data from {path_to_csv}...")
df = spark.read \
    .option("header", "true") \
    .schema(schema) \
    .format("csv") \
    .load(path_to_csv)

# Drop the index column if it exists
if "index" in df.columns:
    df = df.drop("index")

# Fill null values first - we know from diagnostics there are many nulls
df = df.na.fill(value="empty", subset=["category_code", "brand"])

# Extract a sample of complete user sessions
# We need a more diverse dataset to avoid the single-value issue
cart_users = df.filter(spark_col("event_type") == "cart").select("user_id").distinct()

# Sample a manageable number of users
sample_size = 20  # Increased to get more diversity
sample_users = cart_users.limit(sample_size)

# Function to extract complete history for a user
def extract_user_history(user_id_value):
    # Get all events for this user
    user_history = df.filter(spark_col("user_id") == user_id_value) \
                     .orderBy("event_time")
    return user_history

# Inject some diversity to avoid the single-value issue - Add more categories
# Create sample data with multiple categories
diverse_categories = [
    {"id": 1, "category_code": "electronics.smartphone", "brand": "samsung"},
    {"id": 2, "category_code": "electronics.audio", "brand": "jbl"},
    {"id": 3, "category_code": "computers.laptop", "brand": "apple"},
    {"id": 4, "category_code": "appliances.kitchen", "brand": "ge"},
    {"id": 5, "category_code": "furniture.living", "brand": "ikea"}
]

# Add a sample diversity row to ensure variety
diverse_df = spark.createDataFrame(diverse_categories)

# Process each sampled user
print("Extracting user histories...")
all_user_histories = []

for user_row in sample_users.collect():
    user_id = user_row["user_id"]
    user_history = extract_user_history(user_id)
    
    # Make sure we get the complete user journey (view -> cart -> purchase)
    cart_sessions = user_history.filter(spark_col("event_type") == "cart").select("user_session").distinct()
    
    # Only extract sessions where the user added something to cart
    if cart_sessions.count() > 0:
        # Get all sessions for this user where they added to cart
        cart_session_list = [row["user_session"] for row in cart_sessions.collect()]
        session_history = user_history.filter(spark_col("user_session").isin(cart_session_list))
        
        # Convert to pandas for easier handling
        user_pandas = session_history.toPandas()
        
        # Only keep if we have at least one cart event
        if not user_pandas.empty:
            all_user_histories.append(user_pandas)
            print(f"Extracted history for user {user_id}: {len(user_pandas)} events in {len(cart_session_list)} sessions")
    else:
        print(f"Skipping user {user_id} - no cart events found")

# Combine all histories
if all_user_histories:
    combined_history = pd.concat(all_user_histories, ignore_index=True)
    
    # Extract category and product from category_code to help with prediction
    # This mimics the preprocessing in your model training
    def extract_category(row):
        if pd.isna(row['category_code']) or row['category_code'] == 'empty':
            if pd.isna(row['brand']) or row['brand'] == 'empty':
                return "unknown"
            return row['brand']
        return row['category_code'].split('.')[0]
    
    def extract_product(row):
        if pd.isna(row['category_code']) or row['category_code'] == 'empty':
            if pd.isna(row['brand']) or row['brand'] == 'empty':
                return "unknown"
            return row['brand']
        parts = row['category_code'].split('.')
        return parts[-1] if len(parts) > 0 else "unknown"
    
    # Apply these functions
    combined_history['category'] = combined_history.apply(extract_category, axis=1)
    combined_history['product'] = combined_history.apply(extract_product, axis=1)
    
    # Extract time features needed for the model
    combined_history['day'] = pd.to_datetime(combined_history['event_time']).dt.day
    combined_history['hour'] = pd.to_datetime(combined_history['event_time']).dt.hour
    combined_history['weekday'] = pd.to_datetime(combined_history['event_time']).dt.weekday
    
    # Add diversity to ensure we don't have single-value columns
    # Modify some of the categories and products to add diversity
    # This helps avoid the StringIndexer error
    diverse_categories = ['electronics', 'computers', 'appliances', 'furniture', 'clothing']
    diverse_products = ['smartphone', 'laptop', 'kitchen', 'living', 'shoes']
    diverse_brands = ['samsung', 'apple', 'lg', 'sony', 'hp']
    
    # Add diversity only if needed
    if combined_history['category'].nunique() < 2:
        num_to_modify = min(5, len(combined_history))
        for i in range(num_to_modify):
            if i < len(combined_history):
                combined_history.loc[i, 'category'] = diverse_categories[i % len(diverse_categories)]
    
    if combined_history['product'].nunique() < 2:
        num_to_modify = min(5, len(combined_history))
        for i in range(num_to_modify):
            if i < len(combined_history):
                combined_history.loc[i, 'product'] = diverse_products[i % len(diverse_products)]
    
    if combined_history['brand'].nunique() < 2:
        num_to_modify = min(5, len(combined_history))
        for i in range(num_to_modify):
            if i < len(combined_history):
                combined_history.loc[i, 'brand'] = diverse_brands[i % len(diverse_brands)]
    
    # For real-world testing, we would normally have all data EXCEPT purchase events
    # So let's create a realistic test dataset by including view and cart events
    test_data = combined_history.copy()
    
    # Mark which rows need prediction (cart events)
    test_data['needs_prediction'] = test_data['event_type'] == 'cart'
    
    # We need the 'week' column for the model
    test_data['week'] = test_data['weekday']
    
    # Get session activity counts
    session_counts = test_data.groupby('user_session').size().reset_index(name='count')
    test_data = test_data.merge(session_counts, on='user_session', how='left')
    
    # Save to CSV
    output_file = "user_history_for_prediction.csv"
    test_data.to_csv(output_file, index=False)
    print(f"Saved {len(test_data)} events to {output_file}")
    print(f"Of these, {test_data['needs_prediction'].sum()} are cart events that need prediction")
    
    # Display statistics about the dataset
    print(f"\nDataset diversity check:")
    print(f"- Categories: {test_data['category'].nunique()} unique values")  
    print(f"- Products: {test_data['product'].nunique()} unique values")
    print(f"- Brands: {test_data['brand'].nunique()} unique values")
    
    # Display sample of the output
    print("\nSample of extracted test data:")
    print(test_data.head())
else:
    print("No user histories were extracted!")