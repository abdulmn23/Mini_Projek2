import numpy as np
import pandas as pd
import streamlit as st
from essential import get_cached_model, preprocess_input

@st.cache_resource
def load_model_and_scaler():
    """Load and cache model and scaler"""
    try:
        from essential import load_model
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model and scaler with caching
model, scaler = load_model_and_scaler()

def predict_price(input_df):
    """Predict laptop price based on input features"""
    if model is None or scaler is None:
        raise ValueError("Model or scaler not loaded. Please check the model files.")
    
    try:
        if not isinstance(input_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if input_df.empty:
            raise ValueError("Input DataFrame is empty")
        
        data = preprocess_input(input_df, scaler)
        pred = model.predict(data)
        harga = np.exp(pred[0][0])
        
        return round(harga, 2)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise