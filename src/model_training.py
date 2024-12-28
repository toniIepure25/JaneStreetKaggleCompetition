import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

# Paths
KAGGLE_INPUT_DIR = "/data/selected_features"
OUTPUT_DIR = "outputs/"
MODEL_DIR = OUTPUT_DIR + "models"  # Directory to save models
RESULTS_DIR = OUTPUT_DIR + "results"  # Directory to save results
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Data
def load_data():
    print("Loading Data...")
    train_selected_path = os.path.join(KAGGLE_INPUT_DIR, "train_selected_features.parquet")
    train_pca_path = os.path.join(KAGGLE_INPUT_DIR, "train_pca_features.parquet")
    test_selected_path = os.path.join(KAGGLE_INPUT_DIR, "test_selected_features.parquet")

    train_selected_data = pd.read_parquet(train_selected_path)
    train_pca_data = pd.read_parquet(train_pca_path)
    test_selected_data = pd.read_parquet(test_selected_path)
    
    print("Data Loaded Successfully.")
    return train_selected_data, train_pca_data, test_selected_data

# Split Data
def split_data(data, target_column="responder_6", test_size=0.2, random_state=42):
    X = data.drop(columns=[target_column, "weight"])
    y = data[target_column]
    weights = data["weight"]
    return train_test_split(X, y, weights, test_size=test_size, random_state=random_state)

# Train Model
def train_model(model, X_train, y_train, param_grid=None):
    if param_grid:
        print(f"Performing Grid Search for {model.__class__.__name__}...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best Parameters for {model.__class__.__name__}: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        print(f"Training {model.__class__.__name__}...")
        model.fit(X_train, y_train)
        return model

# Evaluate Model
def evaluate_model_weighted(model, X_test, y_test, weights):
    print(f"Evaluating {model.__class__.__name__} with weights...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions, sample_weight=weights)
    r2 = r2_score(y_test, predictions, sample_weight=weights)
    print(f"{model.__class__.__name__} - Weighted MSE: {mse:.4f}, Weighted R²: {r2:.4f}")
    return mse, r2

# Save Results
def save_results(model_name, metrics, dataset_type):
    results_path = os.path.join(RESULTS_DIR, f"{model_name}_{dataset_type}_results.csv")
    pd.DataFrame([metrics]).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

# Save Model
def save_model(model, model_name, dataset_type):
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{dataset_type}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Main Function
def main():
    # Load Data
    train_selected_data, train_pca_data, test_selected_data = load_data()

    # Models and Hyperparameter Grids
    models = {
        "Ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
        "Lasso": (Lasso(), {"alpha": [0.01, 0.1, 1.0]}),
        "RandomForest": (RandomForestRegressor(), {"n_estimators": [100, 300], "max_depth": [5, 10]}),
        "XGBoost": (XGBRegressor(eval_metric='rmsle'), {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1]}),
        "LightGBM": (LGBMRegressor(), {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1]}),
    }

    # Train on Selected Features
    print("\n===== Training on Selected Features Dataset =====")
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(train_selected_data)
    for model_name, (model, param_grid) in models.items():
        best_model = train_model(model, X_train, y_train, param_grid)
        mse, r2 = evaluate_model_weighted(best_model, X_test, y_test, w_test)
        save_results(model_name, {"Weighted MSE": mse, "Weighted R2": r2}, dataset_type="selected_features")
        save_model(best_model, model_name, dataset_type="selected_features")

    # Train on PCA Features
    print("\n===== Training on PCA Features Dataset =====")
    X_train_pca, X_test_pca, y_train_pca, y_test_pca, w_train_pca, w_test_pca = split_data(train_pca_data)
    for model_name, (model, param_grid) in models.items():
        best_model = train_model(model, X_train_pca, y_train_pca, param_grid)
        mse, r2 = evaluate_model_weighted(best_model, X_test_pca, y_test_pca, w_test_pca)
        save_results(model_name, {"Weighted MSE": mse, "Weighted R2": r2}, dataset_type="pca_features")
        save_model(best_model, model_name, dataset_type="pca_features")

if __name__ == "__main__":
    main()



# Best Parameters for LightGBM: {'subsample': 0.7, 'num_leaves': 31, 'n_estimators': 500, 'min_child_samples': 10, 'max_depth': 5, 'learning_rate': 0.05, 'device_type': 'gpu'}
# Model saved to /content/drive/MyDrive/Colab Notebooks/JaneStreetKaggleCompetition/outputs/models/LightGBM_selected_features.joblib
# Evaluating LGBMRegressor...
# LGBMRegressor - MSE: 0.0064, R² Score: 0.9822
# Results saved to /content/drive/MyDrive/Colab Notebooks/JaneStreetKaggleCompetition/outputs/results/LightGBM_selected_features_results.csv
# Predicting on Test Data with LGBMRegressor...
# Test Predictions saved to /content/drive/MyDrive/Colab Notebooks/JaneStreetKaggleCompetition/outputs/results/LightGBM_test_predictions.csv