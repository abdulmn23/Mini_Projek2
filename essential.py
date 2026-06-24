import os
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def get_project_root():
    """Get the project root directory"""
    # In Streamlit Cloud, the app runs from the root of the repo
    if os.environ.get('STREAMLIT_SHARING') or os.environ.get('STREAMLIT_CLOUD'):
        return os.getcwd()
    
    # Local environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find project root by looking for common markers
    markers = ['saved_models', 'data', 'pages']
    search_dir = current_dir
    
    for _ in range(5):
        if all(os.path.exists(os.path.join(search_dir, marker)) for marker in ['saved_models', 'data']):
            return search_dir
        if os.path.exists(os.path.join(search_dir, 'saved_models')):
            return search_dir
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    
    return current_dir

PROJECT_ROOT = get_project_root()

def find_model_file():
    """Try to find model file in different possible locations"""
    possible_paths = [
        os.path.join(PROJECT_ROOT, "saved_models", "ann_model.keras"),
        os.path.join(os.getcwd(), "saved_models", "ann_model.keras"),
        "../saved_models/ann_model.keras",
        "saved_models/ann_model.keras",
        "./saved_models/ann_model.keras",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", "ann_model.keras"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "ann_model.keras"),
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    return None

def find_scaler_file():
    """Try to find scaler file in different possible locations"""
    possible_paths = [
        os.path.join(PROJECT_ROOT, "saved_models", "scaler.joblib"),
        os.path.join(os.getcwd(), "saved_models", "scaler.joblib"),
        "../saved_models/scaler.joblib",
        "saved_models/scaler.joblib",
        "./saved_models/scaler.joblib",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", "scaler.joblib"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "scaler.joblib"),
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    return None

def load_model():
    """Load model and scaler with error handling"""
    model_path = find_model_file()
    scaler_path = find_scaler_file()
    
    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found. Searched in multiple locations.\n"
            f"Project root: {PROJECT_ROOT}\n"
            f"Current directory: {os.getcwd()}"
        )
    
    if scaler_path is None:
        raise FileNotFoundError(
            f"Scaler file not found. Searched in multiple locations.\n"
            f"Project root: {PROJECT_ROOT}\n"
            f"Current directory: {os.getcwd()}"
        )
    
    try:
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        print(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        raise

def preprocess_input(df, scaler):
    """Preprocess input data for prediction"""
    numeric_cols = [
        'Inches', 'Ram', 'Weight', 'Touchscreen', 
        'IPS Panel', 'X_res', 'Y_res', 'HDD', 'SSD'
    ]
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Ensure all columns exist
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Convert to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Select only the required columns in correct order
    df = df[numeric_cols]
    
    # Scale the data
    try:
        scaled = scaler.transform(df)
        return scaled
    except Exception as e:
        print(f"Error in scaling: {e}")
        raise

# Cache for Streamlit
_model_cache = None
_scaler_cache = None

def get_cached_model():
    """Get cached model and scaler for Streamlit"""
    global _model_cache, _scaler_cache
    if _model_cache is None or _scaler_cache is None:
        _model_cache, _scaler_cache = load_model()
    return _model_cache, _scaler_cache