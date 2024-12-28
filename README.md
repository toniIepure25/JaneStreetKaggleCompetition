Jane Street Kaggle Competition: Advanced Feature Engineering and Modeling
This repository showcases a robust feature engineering and modeling pipeline designed for the Jane Street Kaggle Competition, one of the most challenging financial modeling contests. Our solution achieves a sample-weighted zero-mean R² score of 0.978 on validation data, demonstrating its ability to capture complex market dynamics effectively.

Project Overview
The objective of this project is to predict the responder_6 field, a proxy for trading decisions, using a high-dimensional, non-stationary financial dataset. The dataset presents unique challenges:

Fat-tailed distributions and significant noise.
Temporal dependencies and correlations among features.
Complex relationships within lagged responder fields and engineered features.
Our approach combines cutting-edge feature engineering and state-of-the-art machine learning techniques, culminating in an ensemble of XGBoost models and a supervised autoencoder framework to maximize predictive performance.

Approach
1. Data Understanding and Preprocessing
Analyzed the dataset to understand its structure and identify key patterns.
Handled missing values and outliers to ensure data consistency.
Excluded metadata (row_id, date_id, time_id) to focus on predictive features.
2. Feature Engineering
A comprehensive feature engineering pipeline was developed to extract meaningful signals from the data:

Lagged Features:
Added lagged versions of all raw features and responders (responder_0...8) across multiple time horizons (1, 2, 3, 7, 14, 21).
Rolling Statistics:
Computed rolling means and standard deviations for a 7-day window to capture short-term trends.
Exponential Moving Averages (EMA):
Introduced EMA features with spans of 7, 14, and 21 days for momentum-based signals.
Relative Strength Index (RSI):
Incorporated RSI to reflect overbought/oversold conditions.
MACD and Bollinger Bands:
Added trend-following and volatility-based features to enrich the signal space.
Rank Transformations:
Normalized features through rank transformations for improved model robustness.
Feature Interactions:
Generated interaction terms (products and ratios) between key features to model nonlinear dependencies.
Supervised Autoencoders:
Trained autoencoders to learn compressed representations of high-dimensional feature spaces, preserving critical predictive signals while reducing noise.
3. Model Development
Baseline Model:
A simple XGBoost regressor trained on lagged features and rolling statistics achieved a baseline R² score of 0.68.
Advanced Model:
Incorporated the full feature set, including autoencoder-transformed features, into a tuned XGBoost model.
Used hyperparameter optimization (Bayesian search) to refine model performance.
Ensemble Strategy:
Blended predictions from multiple XGBoost models trained on different subsets of features for enhanced generalizability.
4. Evaluation and Iteration
Rigorous validation using out-of-fold predictions and time-based splits.
Reduced overfitting by excluding responder-based features in later iterations, while retaining key predictive signals from engineered features.
Results
Version	Approach	Validation R²
Baseline	Lagged features and rolling statistics	0.680
Intermediate	Added EMA, RSI, and interaction terms	0.890
Final	Full feature set + supervised autoencoder	0.978
The final model consistently demonstrated excellent predictive accuracy on the validation set, capturing complex temporal and cross-feature dependencies.

Key Insights
Responder Fields: Early iterations utilized lagged responder fields (responder_0...8), boosting performance but risking overfitting. Later refinements excluded responders, focusing on raw features and engineered statistics.
Supervised Autoencoders: These provided compact, noise-resistant representations of high-dimensional feature spaces, significantly improving model robustness.
Feature Engineering: Statistical features like rolling stats, EMA, and Bollinger Bands were critical for capturing market behavior.
