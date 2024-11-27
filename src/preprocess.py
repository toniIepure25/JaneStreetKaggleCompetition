import os
import pandas as pd
import numpy as np

# Paths
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
LAGS_FILE = os.path.join(RAW_DATA_DIR, "lags.parquet", "date_id=0", "part-0.parquet")
# TRAIN_FILE = os.path.join(RAW_DATA_DIR, "train", "partition_id=0", "part-0.parquet")
TRAIN_FILE = os.path.join("data", "processed", "combined_train_data.parquet")

def drop_empty_or_constant_columns(df):
    """
    Remove columns that are entirely null or constant.
    """

    non_null_counts = df.notnull().sum()
    constant_columns = df.columns[df.nunique() <= 1]
    drop_column = list(non_null_counts[non_null_counts == 0].index) + list(constant_columns)
    print(f"Dropping {len(drop_column)} columns")
    return df.drop(columns=drop_column)

# def handle_missing_values(df):
#     """
#     Basic imputation for missing values (placeholder logic).
#     """
#     for col in df.columns:
#         if df[col].isnull().sum() > 0:
#             print(f"Imputing missing values in {col}")
#             df[col] = df[col].fillna(df[col].median())
#     return df

def handle_missing_values(df):
    """
    Forward-fill missing values for time-series data.
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"Forward-filling missing values in {col}")
            df[col] = df.groupby('symbol_id')[col].ffill()
    return df

def downcast_dtypes(df):
    """
    Convert columns to more memory-efficient types.
    """
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df

def merge_lags(df, lags_file):
    """
    Merge lagged responder data with training data
    """
    lags = pd.read_parquet(lags_file)
    merged_df = df.merge(lags, on=["date_id", "symbol_id"], how="left")
    return merged_df



def get_responder_columns(df):
    return [col for col in df.columns if col.startswith("responder_")]


  
def preprocess_data(train_file, lags_file):
    """
    Comprehensive preprocessing pipeline for training data.
    """

    print("Loading training data...")
    train_df = pd.read_parquet(train_file)

    # Step 1: Drop empty or constant columns
    print("Step 1: Dropping empty or constant columns...")
    train_df = drop_empty_or_constant_columns(train_df)

    # Step 2: Handle missing values
    print("Step 2: Handling missing values...")
    train_df = handle_missing_values(train_df)

    # Step 3: Optimize memory usage
    print("Step 3: Downcasting data types...")
    train_df = downcast_dtypes(train_df)

    #! Merge or not
    # Step 4: Merge lagged data
    print("Step 4: Merging lagged responder data...")
    train_df = merge_lags(train_df, lags_file)

    # Step 16: Save the processed data
    print("Step 5: Saving processed data...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, "train_processed.parquet")
    train_df.to_parquet(processed_file_path, index=False)
    print(f"Processed train data saved to {processed_file_path}")
    
if __name__ == "__main__":
    print("Preprocessing data...")
    preprocess_data(TRAIN_FILE, LAGS_FILE)

