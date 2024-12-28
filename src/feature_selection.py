import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("data", "selected_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function: Load and Preprocess Data
def load_and_preprocess(data_path, target_column, weight_column, test_size=0.1, random_state=42):
    print("Loading and Preprocessing Data...")
    data = pd.read_parquet(data_path)
    data.fillna(data.mean(), inplace=True)

    features = [
        col for col in data.columns
        if col not in [target_column, weight_column]
        and not col.startswith(f"{target_column}_")
        and not col.endswith(f"_{target_column}")
    ] + [col for col in data.columns if f"{target_column}_lag_" in col]

    X = data[features]
    y = data[target_column]
    weights = data[weight_column]

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data Splitting and Scaling Completed.")
    return X_train, X_train_scaled, y_train, weights_train, X_test, X_test_scaled, y_test, weights_test, features

# Function: Correlation-Based Feature Selection
def correlation_feature_selection(X, y, features, threshold=0.1):
    print("Performing Correlation-Based Feature Selection...")
    data = pd.DataFrame(X, columns=features)
    data['target'] = y

    correlation = data.corr()['target'].drop('target')
    selected_features = correlation[abs(correlation) > threshold].index.tolist()
    print(f"Selected {len(selected_features)} features using correlation.")
    return selected_features

# Function: Univariate Feature Selection (F-Test)
def univariate_feature_selection(X, y, features, k=50):
    print("Performing Univariate Feature Selection...")
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    selected_features = np.array(features)[selector.get_support()].tolist()
    print(f"Selected {len(selected_features)} features using univariate F-Test.")
    return selected_features

# Function: Random Forest Feature Importance
def random_forest_feature_selection(X, y, features, top_n=50):
    print("Performing Feature Selection using Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=50,  # Reduced for faster processing
        max_depth=10,  # Limit depth for computational efficiency
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    rf.fit(X, y)

    feature_importances = rf.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    selected_features = importance_df.head(top_n)['Feature'].tolist()
    print(f"Selected {len(selected_features)} features using Random Forest.")
    return selected_features

# Function: PCA for Dimensionality Reduction
def pca_feature_selection(X_scaled, explained_variance=0.95):
    print("Performing Dimensionality Reduction using PCA...")
    pca = PCA(n_components=explained_variance)
    pca_transformed = pca.fit_transform(X_scaled)

    print(f"Number of Components Retained: {pca.n_components_}")
    return pca_transformed, [f"PCA_Component_{i}" for i in range(pca.n_components_)]

# Function: Save Selected Features
def save_selected_features(data, selected_features, target_column, weight_column, train_indices, test_indices, train_output_file, test_output_file):
    print("Saving Final Datasets with Selected Features...")
    train_data = data.iloc[train_indices][selected_features + [target_column, weight_column]]
    test_data = data.iloc[test_indices][selected_features + [target_column, weight_column]]

    train_data.to_parquet(train_output_file, index=False)
    test_data.to_parquet(test_output_file, index=False)

    print(f"Train Dataset Saved to: {train_output_file}")
    print(f"Test Dataset Saved to: {test_output_file}")

# Main Function
def main():
    DATA_FILE = os.path.join(DATA_DIR, "train_autoencoded.parquet")
    TARGET_COLUMN = "responder_6"
    WEIGHT_COLUMN = "weight"
    TRAIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_selected_features.parquet")
    TEST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_selected_features.parquet")

    data = pd.read_parquet(DATA_FILE)
    (
        X_train, X_train_scaled, y_train, weights_train,
        X_test, X_test_scaled, y_test, weights_test, features
    ) = load_and_preprocess(DATA_FILE, TARGET_COLUMN, WEIGHT_COLUMN)

    train_indices = X_train.index
    test_indices = X_test.index

    corr_features = correlation_feature_selection(X_train, y_train, features, threshold=0.1)
    univariate_features = univariate_feature_selection(X_train_scaled, y_train, features, k=50)
    rf_features = random_forest_feature_selection(X_train_scaled, y_train, features, top_n=50)

    final_selected_features = list(set(corr_features + univariate_features + rf_features))
    print(f"Total Combined Selected Features: {len(final_selected_features)}")

    save_selected_features(
        data,
        final_selected_features,
        TARGET_COLUMN,
        WEIGHT_COLUMN,
        train_indices,
        test_indices,
        TRAIN_OUTPUT_FILE,
        TEST_OUTPUT_FILE
    )

if __name__ == "__main__":
    main()
