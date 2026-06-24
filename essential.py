import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


MODEL_PATH = "../saved_models/ann_model.keras"
SCALER_PATH = "../saved_models/scaler.joblib"


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def preprocess_input(df, scaler):

    numeric_cols = [
        'Inches',
        'Ram',
        'Weight',
        'Touchscreen',
        'IPS Panel',
        'X_res',
        'Y_res',
        'HDD',
        'SSD'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    scaled = scaler.transform(df)

    return scaled