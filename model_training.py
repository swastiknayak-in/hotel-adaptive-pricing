# model_training.py
# Loads the dataset, preprocesses it, trains two models (LinearRegression & RandomForest),
# selects the best (RMSE), and saves the model + preprocessor to models/.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


def load_data(path='data/hotel_bookings.csv'):
    df = pd.read_csv(path)
    return df


def _clean_month(col):
    # Convert month names to numeric month (jan=1 ... dec=12)
    months = {
        'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
        'July':7,'August':8,'September':9,'October':10,'November':11,'December':12
    }
    return col.map(months).fillna(col)


def preprocess_features(df):
    df = df.copy()

    # Ensure expected columns exist
    expected_cols = ['hotel','lead_time','arrival_date_month','reserved_room_type',
                     'assigned_room_type','customer_type','previous_bookings_not_canceled','adr']
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in dataset")

    # Keep only needed columns
    df = df[expected_cols]

    # Convert month name to numeric
    df['arrival_month'] = _clean_month(df['arrival_date_month'])
    df['arrival_month'] = df['arrival_month'].astype(int)

    # Drop rows with missing target
    df = df.dropna(subset=['adr'])

    # Features and target
    X = df[['hotel','lead_time','arrival_month','reserved_room_type','customer_type','previous_bookings_not_canceled']]
    y = df['adr'].astype(float)

    return X, y


def train_and_save(data_path='data/hotel_bookings.csv', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    df = load_data(data_path)
    X, y = preprocess_features(df)

    # Simple preprocessing: numeric vs categorical
    numeric_features = ['lead_time','arrival_month','previous_bookings_not_canceled']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['hotel','reserved_room_type','customer_type']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
    train_and_save()