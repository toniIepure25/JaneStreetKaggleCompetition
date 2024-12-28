import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from itertools import combinations
from scipy.fft import fft

# Paths
PROCESSED_DATA_DIR = os.path.join("data", "processed")
TRAIN_PREPROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "train_preprocessed.parquet")

def add_custom_lags(df, cols, lags):
    """
    Adds lagged features for specified columns.
    """
    print("Step 1: Adding custom lagged features...")
    for col in cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df.groupby("symbol_id")[col].shift(lag)
    print(f" - Added {len(cols) * len(lags)} lagged features.")
    return df

def add_rolling_features(df, feature_cols, window):
    """
    Adds rolling mean and standard deviation.
    """
    print(f"Step 2: Adding rolling features (window={window})...")
    for col in feature_cols:
        df[f"{col}_rolling_mean_{window}"] = df.groupby("symbol_id")[col].transform(lambda x: x.rolling(window).mean())
        df[f"{col}_rolling_std_{window}"] = df.groupby("symbol_id")[col].transform(lambda x: x.rolling(window).std())
    return df

def add_ema_features(df, cols, spans):
    """
    Adds Exponential Moving Averages (EMA) for specified columns.
    """
    print("Step 3: Adding EMA features...")
    for col in cols:
        for span in spans:
            df[f"{col}_ema_{span}"] = df.groupby("symbol_id")[col].transform(lambda x: x.ewm(span=span).mean())
    return df

def add_fft_features(df, cols, n_components=5, exclude_col="responder_6"):
    """
    Adds FFT (Fast Fourier Transform) components for specified columns while avoiding leakage.
    """
    print("Step 4: Adding FFT features...")
    fft_features = {}

    for col in cols:
        if col == exclude_col:
            print(f" - Skipping FFT for excluded column: {col}")
            continue

        print(f" - Processing FFT for {col}...")
        def compute_fft(x):
            x_shifted = x.shift(1).fillna(0).to_numpy()
            fft_result = np.abs(fft(x_shifted))[:n_components]
            return fft_result

        grouped = df.groupby("symbol_id")[col].apply(compute_fft)

        for i in range(n_components):
            fft_features[f"{col}_fft_{i+1}"] = grouped.apply(lambda x: x[i] if len(x) > i else np.nan)

    fft_features_df = pd.DataFrame(fft_features)
    print(f" - Added FFT features for {len(cols) - 1 if exclude_col in cols else len(cols)} columns with {n_components} components each.")
    return pd.concat([df.reset_index(drop=True), fft_features_df.reset_index(drop=True)], axis=1)

def add_rsi(df, col, window=14):
    """
    Adds Relative Strength Index (RSI) for a specified column.
    Excludes the current value to avoid data leakage.
    """
    print(f"Adding RSI for {col}...")
    delta = df[col].diff().shift(1)
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-5)
    df[f"{col}_rsi"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, col, fast_span=12, slow_span=26, signal_span=9):
    """
    Adds MACD features for a specified column.
    Uses only past values to avoid data leakage.
    """
    print(f"Adding MACD for {col}...")
    fast_ema = df[col].ewm(span=fast_span, adjust=False).mean().shift(1)
    slow_ema = df[col].ewm(span=slow_span, adjust=False).mean().shift(1)
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    df[f"{col}_macd"] = macd
    df[f"{col}_macd_signal"] = signal
    return df

def add_bollinger_bands(df, col, window=20, num_std_dev=2):
    """
    Adds Bollinger Bands for a specified column.
    Uses only historical data to avoid leakage.
    """
    print(f"Adding Bollinger Bands for {col}...")
    rolling_mean = df[col].rolling(window=window, min_periods=1).mean().shift(1)
    rolling_std = df[col].rolling(window=window, min_periods=1).std().shift(1)
    df[f"{col}_bollinger_upper"] = rolling_mean + (rolling_std * num_std_dev)
    df[f"{col}_bollinger_lower"] = rolling_mean - (rolling_std * num_std_dev)
    return df

def add_atr(df, high_col, low_col, close_col, window=14):
    """
    Adds Average True Range (ATR) for volatility measurement.
    """
    print(f"Adding ATR for {close_col}...")
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift(1))
    low_close = np.abs(df[low_col] - df[close_col].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"{close_col}_atr"] = tr.rolling(window=window, min_periods=1).mean()
    return df

def add_drawdown(df, col):
    """
    Adds drawdown percentage for a specified column.
    Excludes the current value to prevent leakage.
    """
    print(f"Adding Drawdown for {col}...")
    cumulative_max = df[col].cummax().shift(1)
    df[f"{col}_drawdown"] = (df[col].shift(1) - cumulative_max) / cumulative_max
    return df

def add_interaction_features(df, cols):
    """
    Adds interaction terms (products and ratios) between specified columns.
    """
    print("Step 5: Adding interaction features...")
    for col1, col2 in combinations(cols, 2):
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-5)
    return df

def add_rank_transform(df, cols):
    """
    Adds rank-transformed versions of specified columns.
    """
    print("Adding Rank Transformation...")
    for col in cols:
        df[f"{col}_rank"] = df[col].rank() / len(df)
    return df

def feature_engineering(train_file, output_file):
    """
    Feature engineering pipeline.
    """
    print("Loading preprocessed data...")
    df = pd.read_parquet(train_file)
    responders = [col for col in df.columns if col.startswith("responder_")]

    df = add_custom_lags(df, responders, lags=[1, 2, 3, 7, 14, 21])
    df = add_rolling_features(df, responders, window=7)
    df = add_ema_features(df, responders, spans=[7, 14, 21])
    df = add_fft_features(df, responders, n_components=5, exclude_col="responder_6")

    for col in responders:
        df = add_rsi(df, col)
        df = add_macd(df, col)
        df = add_bollinger_bands(df, col)
        df = add_drawdown(df, col)

    df = add_rank_transform(df, responders)
    df = add_interaction_features(df, responders)

    print("Saving feature-engineered data...")
    df.to_parquet(output_file, index=False)
    print(f"Feature-engineered data saved to {output_file}")

if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_file = os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet")
    feature_engineering(TRAIN_PREPROCESSED_FILE, output_file)
