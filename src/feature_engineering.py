def add_higher_moment_features(df):
    """
    Adds skewness and kurtosis features to the dataset.
    """
    df['kurtosis'] = df.groupby('symbol_id')['responder_6'].transform(lambda x: kurtosis(x, fisher=False))
    df['skewness'] = df.groupby('symbol_id')['responder_6'].transform(lambda x: skew(x))
    return df

