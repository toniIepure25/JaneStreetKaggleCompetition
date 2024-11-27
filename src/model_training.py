from sklearn.linear_model import HuberRegressor
model = HuberRegressor()
model.fit(X_train, y_train)