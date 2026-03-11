# model_training.py
# Train hotel price prediction model and save it

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# -----------------------------
# Load Dataset
# -----------------------------
def load_data(path="data/hotel_bookings.csv"):
    df = pd.read_csv(path)
    return df


# -----------------------------
# Convert Month Name → Number
# -----------------------------
def clean_month(col):

    months = {
        "January":1,"February":2,"March":3,"April":4,
        "May":5,"June":6,"July":7,"August":8,
        "September":9,"October":10,"November":11,"December":12
    }

    return col.map(months)


# -----------------------------
# Feature Preprocessing
# -----------------------------
def preprocess_features(df):

    df = df.copy()

    required_columns = [
        "hotel",
        "lead_time",
        "arrival_date_month",
        "reserved_room_type",
        "customer_type",
        "previous_bookings_not_canceled",
        "adr"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Convert month name to numeric
    df["arrival_month"] = clean_month(df["arrival_date_month"])

    df = df.dropna(subset=["adr"])

    X = df[
        [
            "hotel",
            "lead_time",
            "arrival_month",
            "reserved_room_type",
            "customer_type",
            "previous_bookings_not_canceled"
        ]
    ]

    y = df["adr"]

    return X, y


# -----------------------------
# Train Model
# -----------------------------
def train_and_save(data_path="data/hotel_bookings.csv", model_dir="models"):

    os.makedirs(model_dir, exist_ok=True)

    df = load_data(data_path)

    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Numeric features
    numeric_features = [
        "lead_time",
        "arrival_month",
        "previous_bookings_not_canceled"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical features
    categorical_features = [
        "hotel",
        "reserved_room_type",
        "customer_type"
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Models to test
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print(f"{name} RMSE:", rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

    # Save model
    model_path = os.path.join(model_dir, "pricing_model.pkl")

    joblib.dump(best_model, model_path)

    print("Model saved at:", model_path)

    return best_model


# -----------------------------
# Run training
# -----------------------------
if __name__ == "__main__":
    train_and_save()
