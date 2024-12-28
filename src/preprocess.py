import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

# Paths
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, "combined_train_data.parquet")

def split_long_tailed_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets while preserving long-tailed distribution.
    """
    print("Step 0: Splitting data into train and test sets...")
    df['target_binned'] = pd.qcut(df[target_column], q=100, duplicates='drop', labels=False)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['target_binned'],
        random_state=random_state
    )
    train_df.drop(columns=['target_binned'], inplace=True)
    test_df.drop(columns=['target_binned'], inplace=True)
    print(f"Data split completed: Train size = {len(train_df)}, Test size = {len(test_df)}")
    return train_df, test_df

def drop_empty_or_constant_columns(df):
    """
    Remove columns that are entirely null or constant.
    """
    print("Step 1: Dropping empty or constant columns...")
    constant_cols = df.columns[df.nunique() <= 1]
    print(f" - Dropped {len(constant_cols)} constant or empty columns.")
    return df.drop(columns=constant_cols)

def handle_missing_values(df, critical_cols, other_cols, n_neighbors=5):
    """
    Handles missing values using KNN for critical columns and forward-fill for others.
    """
    print("Step 2: Handling missing values...")
    print(f" - Applying KNN Imputer for {len(critical_cols)} critical columns...")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[critical_cols] = imputer.fit_transform(df[critical_cols])
    print(f" - Forward-filling for {len(other_cols)} other columns...")
    df[other_cols] = df[other_cols].ffill()
    return df

def downcast_dtypes(df):
    """
    Downcasts numerical data types to optimize memory usage.
    """
    print("Step 3: Downcasting numerical data types...")
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    print(" - Downcasting complete.")
    return df

def preprocess_data(train_file, output_dir):
    """
    Preprocessing pipeline for data cleaning and preparation.
    """
    print("Loading training data...")
    df = pd.read_parquet(train_file)
    responders = [col for col in df.columns if col.startswith("responder_")]

    # Step 0: Split into train and test sets
    train_df, test_df = split_long_tailed_data(df, target_column="responder_6")

    for subset_name, subset_df in [("train", train_df), ("test", test_df)]:
        print(f"Preprocessing {subset_name} data...")
        subset_df = drop_empty_or_constant_columns(subset_df)
        critical_cols = [col for col in subset_df.columns if col in responders]
        other_cols = [col for col in subset_df.columns if col not in critical_cols]
        subset_df = handle_missing_values(subset_df, critical_cols, other_cols)
        subset_df = downcast_dtypes(subset_df)
        output_file = os.path.join(output_dir, f"{subset_name}_preprocessed.parquet")
        subset_df.to_parquet(output_file, index=False)
        print(f"{subset_name.capitalize()} data saved to {output_file}")

if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    preprocess_data(TRAIN_FILE, PROCESSED_DATA_DIR)
