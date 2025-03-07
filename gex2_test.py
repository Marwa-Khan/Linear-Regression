from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from gex2 import setup_preprocessor, simple_linear_regression, multiple_linear_regression, calculate_metrics

# Load the dataset once for all tests
@pytest.fixture(scope="module")
def dataset():
    
    file_path = "test_dataset.csv"
    df = pd.read_csv(file_path)
    return df

@pytest.fixture(autouse=True)
def suppress_plots():
    with patch("matplotlib.pyplot.show"):
        yield

def test_calculate_metrics():
    
    y_true = np.random.randint(10, 100, size=10)
    y_pred = y_true + np.random.randint(-5, 5, size=10)

    mse, rmse, mae, r2 = calculate_metrics(y_true, y_pred)

    assert mse >= 0, "MSE should always be non-negative"
    assert rmse >= 0, "RMSE should always be non-negative"
    assert mae >= 0, "MAE should always be non-negative"
    assert -1 <= r2 <= 1, "R-squared value should be between -1 and 1"

def test_simple_linear_regression_train_test(dataset):
    
    X = dataset[["Square_Feet"]]
    y = dataset["Price"]

    simple_linear_regression(X, y, "train-test")

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse, _, _, _ = calculate_metrics(y_test, y_pred)

    
    assert mse > 0  
    assert model.coef_[0] != 0  

def test_simple_linear_regression_k_fold(dataset):
    
    X = dataset[["Square_Feet"]]
    y = dataset["Price"]

    k = 5
    simple_linear_regression(X, y, "k-fold", k=k)

    
    model = Pipeline(steps=[('preprocessor', setup_preprocessor(X)), ('regressor', LinearRegression())])
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(mse_scores)

    
    assert len(mse_scores) == k
    assert np.mean(mse_scores) > 0
    assert np.mean(rmse_scores) > 0


def test_multiple_linear_regression_train_val_test(dataset):
    
    X = dataset[["Square_Feet", "Location_Score"]]
    y = dataset["Price"]

    multiple_linear_regression(X, y, "train-val-test")

    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    mse_val = mean_squared_error(y_val, y_val_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    
    assert mse_val > 0
    assert mse_test > 0

def test_multiple_linear_regression_k_fold(dataset):
    
    X = dataset[["Square_Feet", "Location_Score"]]
    y = dataset["Price"]

    k = 5
    multiple_linear_regression(X, y, "k-fold", k=k)

    
    model = Pipeline(steps=[('preprocessor', setup_preprocessor(X)), ('regressor', LinearRegression())])
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)

    
    assert len(mse_scores) == k
    assert np.mean(mse_scores) > 0
    assert np.mean(r2_scores) > 0

if __name__ == "__main__":
    pytest.main()
