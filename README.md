# Jane Street Real-Time Market Data Forecasting

## Overview
This project focuses on forecasting financial market responders using anonymized real-world data derived from production systems. The competition highlights challenges in modeling financial markets, including:
- Fat-tailed distributions
- Non-stationary time series
- Sudden shifts in market behavior

The primary objective is to predict the target variable, `responder_6`, for up to six months into the future.

## Dataset
The dataset includes:
- **train.parquet**: Historical training data with 79 anonymized features and 9 responders.
- **test.parquet**: Mock test set structure for evaluation.
- **lags.parquet**: Lagged values of responders for each `date_id`.
- **features.csv**: Metadata for features.
- **responders.csv**: Metadata for responders.
- **sample_submission.csv**: Example submission format.

## Project Structure
