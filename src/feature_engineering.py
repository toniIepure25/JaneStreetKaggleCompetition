import os
import pandas as pd
import numpy as np


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
    Add summary statistics (e.g., mean, std) over lagged features.

    Args:
        df (pd.DataFrame): The DataFrame with lagged features.
        cols (list): List of base column names (e.g., "responder_0").
        lags (list): List of lag periods (e.g., [1, 2, 3]).

    Returns:
        pd.DataFrame: The DataFrame with aggregated lag features added.
    """
    aggregated_features = {}

    for col in cols:
        # Construct lag column names
        lag_cols = [f'{col}_lag_{lag}' for lag in lags if f'{col}_lag_{lag}' in df.columns]
        
        if not lag_cols:
            print(f"Warning: No valid lag columns found for base column '{col}'.")
            continue

        # Calculate aggregates
        aggregated_features[f'{col}_lag_mean'] = df[lag_cols].mean(axis=1)
        aggregated_features[f'{col}_lag_std'] = df[lag_cols].std(axis=1)
    
    # Create DataFrame for aggregated features
    aggregated_features_df = pd.DataFrame(aggregated_features)
    
    # Concatenate with original DataFrame
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

def feature_engineering(train_file):

    print("Loading training data...")
    train_df = pd.read_parquet(train_file)
    
    print("Step 1: Adding custom lags...")
    responders = get_responder_columns(train_df)
    #train_df = add_custom_lags(train_df, responders, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21])
    train_df = add_custom_lags(train_df, responders, lags=[2, 3, 4, 5, 6, 7, 14, 21])

    # Step 2: Create rolling statistics
    print("Step 2: Adding rolling features...")
    train_df = add_rolling_features(train_df, responders, window=7)

    # Step 3: Add cyclical features for periodicity
    print("Step 3: Adding cyclical features...")
    time_id_col = next((col for col in ["time_id", "time_id_x", "time_id_y"] if col in train_df.columns), None)
    if time_id_col is None:
        raise KeyError("No valid 'time_id' column found in the dataset.")
    train_df = add_cyclical_features(train_df, "date_id", max_value=train_df[time_id_col].max())

    # Step 4: Standardize numerical features
    print("Step 4: Standardizing features...")
    train_df = standardize_features(train_df, responders)

    # Step 5: Normalize numerical features
    print("Step 5: Normalizing features...")
    train_df = normalize_features(train_df, responders)

    # Step 6: Add lag aggregates
    print("Step 6: Adding lag aggregates...")
    train_df = add_lag_aggregates(train_df, responders, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21])

    # Step 7: Add Exponential Moving Averages (EMA)
    print("Step 7: Adding EMA features...")
    ema_spans = [7, 14, 21, 50]
    for span in ema_spans:
        train_df = add_ema_features(train_df, responders, span=span)

    # Step 8: Add time-based features
    print("Step 8: Adding time features...")
    train_df = add_time_features(train_df, 'date_id')

    # Step 9: Quantile transformation
    print("Step 9: Quantile transforming features...")
    train_df = quantile_transform_features(train_df, responders)

    # Step 14: Add interaction features
    # print("Step 14: Adding interaction features...")
    # train_df = add_interaction_features(train_df, [('responder_1', 'responder_2'), ('responder_6', 'responder_7')])

    # Step 15: Clip outliers
    # print("Step 15: Clipping outliers...")
    # train_df = clip_outliers(train_df, responders, lower_percentile=1, upper_percentile=99)

    # Step 10: Save the processed data
    print("Step 10: Saving engineered data...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet")
    train_df.to_parquet(processed_file_path, index=False)
    print(f"Engineered train data saved to {processed_file_path}")


PROCESSED_DATA_DIR = os.path.join("data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "combined_train_data.parquet")
if __name__ == '__main__':
    feature_engineering(PROCESSED_FILE)