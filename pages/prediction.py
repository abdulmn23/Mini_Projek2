import numpy as np
import pandas as pd
import streamlit as st
from essential import get_cached_model, preprocess_input

# Use cached model for Streamlit
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
    """
    Predict laptop price based on input features
    
    Parameters:
    input_df (pd.DataFrame): DataFrame containing laptop features
    
    Returns:
    float: Predicted price
    """
    if model is None or scaler is None:
        raise ValueError("Model or scaler not loaded. Please check the model files.")
    
    try:
        # Validate input
        if not isinstance(input_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if input_df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Preprocess the input
        data = preprocess_input(input_df, scaler)
        
        # Make prediction
        pred = model.predict(data)
        
        # Transform back from log scale
        harga = np.exp(pred[0][0])
        
        return round(harga, 2)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise

# Test function
if __name__ == "__main__":
    # Create test data
    test_df = pd.DataFrame({
        'Inches': [15.6],
        'Ram': [8],
        'Weight': [2.5],
        'Touchscreen': [0],
        'IPS Panel': [1],
        'X_res': [1920],
        'Y_res': [1080],
        'HDD': [500],
        'SSD': [256]
    })
    
    try:
        price = predict_price(test_df)
        print(f"Predicted price: Rp {price:,.2f}")
    except Exception as e:
        print(f"Test failed: {e}")