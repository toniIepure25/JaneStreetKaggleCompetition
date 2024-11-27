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

def add_custom_lags(df, cols, lags):
    """
    Adds custom lagged features for specified columns efficiently.
    """
    lagged_features = {}

    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        
        for lag in lags:
            lagged_features[f'{col}_lag_{lag}'] = (
                df.groupby('symbol_id')[col].shift(lag)
            )
    
    lagged_features_df = pd.DataFrame(lagged_features)
    return pd.concat([df.reset_index(drop=True), lagged_features_df.reset_index(drop=True)], axis=1)

def add_rolling_features(df, feature_cols, window=7):
    # Initialize a dictionary to hold the rolling features
    rolling_features = {}
    
    # Loop through each feature column to calculate rolling mean and std
    for col in feature_cols:
        # Calculate rolling mean and std per symbol_id group
        rolling_mean = df.groupby('symbol_id')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        rolling_std = df.groupby('symbol_id')[col].transform(lambda x: x.rolling(window, min_periods=1).std())
        
        # Add these rolling statistics to the dictionary with unique column names
        rolling_features[f"{col}_rolling_mean_{window}"] = rolling_mean
        rolling_features[f"{col}_rolling_std_{window}"] = rolling_std
    
    # Convert the rolling features dictionary into a DataFrame
    rolling_features_df = pd.DataFrame(rolling_features)
    
    # Concatenate the original DataFrame with the rolling features DataFrame
    return pd.concat([df, rolling_features_df], axis=1)

def add_cyclical_features(df, column, max_value):
    """
    Encode cyclical features using sine and cosine transformations.
    """
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df

def clip_outliers(df, columns, lower_percentile=1, upper_percentile=99):
    """
    Clip outliers based on percentiles.
    """
    for col in columns:
        lower = np.percentile(df[col], lower_percentile)
        upper = np.percentile(df[col], upper_percentile)
        df[col] = np.clip(df[col], lower, upper)
    return df

def add_ema_features(df, cols, span=7):
    """
    Add Exponential Moving Averages (EMA) for specified columns.
    """
    ema_features = {}

    for col in cols:
        # Calculate EMA and ensure it's a 1D Series
        ema = df.groupby('symbol_id')[col].transform(lambda x: x.ewm(span=span, adjust=False).mean())
        ema_features[f'{col}_ema_{span}'] = ema.reset_index(drop=True)  # Flatten to 1D Series
    
    # Convert the dictionary to a DataFrame and concatenate with the original DataFrame
    ema_features_df = pd.DataFrame(ema_features)
    return pd.concat([df.reset_index(drop=True), ema_features_df.reset_index(drop=True)], axis=1)


def logarithmic_transform(df):
    """
    Apply logarithmic transformation to selected columns
    """

    # TODO
    return df

def standardize_features(df, feature_cols):
    """
    Standardizes specified columns of a DataFrame (zero mean, unit variance).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        feature_cols (list of str): List of column names to standardize.

    Returns:
        pd.DataFrame: The DataFrame with standardized columns added.
    """
    for col in feature_cols:
        if col in df.columns:
            # Extract column and ensure it is 1D
            column_data = df[col]
            if isinstance(column_data, pd.DataFrame):
                # Handle case where column selection results in multiple columns
                print(f"Warning: Column '{col}' has duplicates or is 2D. Taking the first match.")
                column_data = column_data.iloc[:, 0]
            
            # Perform standardization
            df[f'{col}_standardized'] = (column_data - column_data.mean()) / column_data.std()
        else:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")
    return df




def normalize_features(df, feature_cols):
    """
    Normalizes the specified columns of a DataFrame to the range [0, 1].

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        feature_cols (list of str): List of column names to normalize.

    Returns:
        pd.DataFrame: The DataFrame with normalized columns added.
    """
    for col in feature_cols:
        if col in df.columns:
            # Extract column and ensure it is 1D
            column_data = df[col]
            if isinstance(column_data, pd.DataFrame):
                print(f"Warning: Column '{col}' has duplicates or is 2D. Taking the first match.")
                column_data = column_data.iloc[:, 0]  # Take the first column

            # Perform normalization
            min_val, max_val = column_data.min(), column_data.max()
            normalized = (column_data - min_val) / (max_val - min_val)

            # Ensure normalized is 1D
            if isinstance(normalized, pd.DataFrame):
                normalized = normalized.iloc[:, 0]

            df[f'{col}_normalized'] = normalized.reset_index(drop=True)
        else:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")
    return df




def add_lag_aggregates(df, cols, lags):
    """
    Add summary statistics (e.g., mean, max, min) over lagged features.
    """
    aggregated_features = {}
    
    for col in cols:
        lag_cols = [f'{col}_lag_{lag}' for lag in lags]
        aggregated_features[f'{col}_lag_mean'] = df[lag_cols].mean(axis=1)
        aggregated_features[f'{col}_lag_std'] = df[lag_cols].std(axis=1)
    
    aggregated_features_df = pd.DataFrame(aggregated_features)
    return pd.concat([df.reset_index(drop=True), aggregated_features_df.reset_index(drop=True)], axis=1)

def add_time_features(df, date_col):
    """
    Add time-based features such as day of the week, month, or year.
    """
    df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

from sklearn.preprocessing import QuantileTransformer

def quantile_transform_features(df, feature_cols, output_distribution='uniform'):
    """
    Apply quantile transformation to features for normalization.
    """
    transformer = QuantileTransformer(output_distribution=output_distribution, random_state=42)
    transformed_cols = transformer.fit_transform(df[feature_cols])
    
    for i, col in enumerate(feature_cols):
        df[f'{col}_quantile'] = transformed_cols[:, i]
    return df

def add_interaction_features(df, feature_pairs):
    """
    Add interaction terms for specified feature pairs.
    """
    for col1, col2 in feature_pairs:
        interaction_col = f'{col1}_x_{col2}'
        df[interaction_col] = df[col1] * df[col2]
    return df


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

