import os
import pandas as pd
import numpy as np

# Paths
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
LAGS_FILE = os.path.join(RAW_DATA_DIR, "lags.parquet", "date_id=0", "part-0.parquet")
TRAIN_FILE = os.path.join(RAW_DATA_DIR, "train.parquet", "partition_id=0", "part-0.parquet")

def drop_empty_or_constant_columns(df):
    """
    Remove columns that are entirely null or constant.
    """

    non_null_counts = df.notnull().sum()
    constant_columns = df.columns[df.nunique() <= 1]
    drop_column = list(non_null_counts[non_null_counts == 0].index) + list(constant_columns)
    print(f"Dropping {len(drop_column)} columns")
    return df.drop(columns=drop_column)

def handle_missing_values(df):
    """
    Basic imputation for missing values (placeholder logic)
    """
    # Fill missing values with column median as a placeholder
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"Imputing missing values in {col}")
            df[col].fillna(df[col].median(), inplace=True)
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

def preprocess_data(train_file, lags_file):
    """
    Preprocess data
    """

    # Load raw train data
    training_data = pd.read_parquet(TRAIN_FILE)

    # Step 1: Drop empty or constant columns
    train_df = drop_empty_or_constant_columns(training_data)

    # Step 2: Handle missing values
    train_df = handle_missing_values(train_df)

    # Step 3: Downcast data types
    train_df = downcast_dtypes(train_df)

    # Step 4: Merge lagged responder data
    train_df = merge_lags(train_df, LAGS_FILE)

    # Step 5: Save processed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, "train_processed.parquet")
    train_df.to_parquet(processed_file_path, index=False)
    print(f"Processed train data saved to {processed_file_path}")

if __name__ == "__main__":
    preprocess_data(TRAIN_FILE, LAGS_FILE)