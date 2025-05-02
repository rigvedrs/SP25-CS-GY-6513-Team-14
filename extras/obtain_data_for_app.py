import pandas as pd

# From the original code, we can see that the following columns are selected for features:
# features = df_targets_week.select("event_type", "brand", "price", "count", "week", "category", "product", "is_purchased")
# The "is_purchased" column is derived from "event_type" using the is_purchased_label UDF

# Load the click stream data
try:
    # Try to load the data - adapt the filename as needed
    df = pd.read_csv('small_oct_2019.csv')
    print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if we have the necessary columns
    if 'event_type' in df.columns:
        # Create a copy of the dataframe without the event_type column
        # event_type is what's used to determine if an item was purchased
        features_only = df.drop(columns=['event_type'])
        
        # Save features only (without the target indicator)
        features_only.to_csv('features_only.csv', index=False)
        print(f"Created features_only.csv with {features_only.shape[0]} rows and {features_only.shape[1]} columns")
        print("This file contains all data except the event_type column which is used to derive the target variable")
    else:
        print("Error: 'event_type' column not found in the data")

except Exception as e:
    print(f"Error processing data: {e}")
    print("Please ensure the data file is in the correct location and format")