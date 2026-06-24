import os
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def get_project_root():
    """Get the project root directory"""
    # Try to find project root by looking for common markers
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're in a subdirectory
    markers = ['saved_models', 'data', 'pages']
    
    # Start from current directory and go up until we find markers
    search_dir = current_dir
    for _ in range(5):  # Go up max 5 levels
        # Check if this directory contains project markers
        if all(os.path.exists(os.path.join(search_dir, marker)) for marker in ['saved_models', 'data']):
            return search_dir
        # Also check if saved_models exists
        if os.path.exists(os.path.join(search_dir, 'saved_models')):
            return search_dir
        # Go up one level
        parent = os.path.dirname(search_dir)
        if parent == search_dir:  # Reached root
            break
        search_dir = parent
    
    # If we can't find markers, use the directory containing this file
    return current_dir

# Get project root
PROJECT_ROOT = get_project_root()

# Build paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "ann_model.keras")
SCALER_PATH = os.path.join(PROJECT_ROOT, "saved_models", "scaler.joblib")

def find_model_file():
    """Try to find model file in different possible locations"""
    possible_paths = [
        MODEL_PATH,
        "../saved_models/ann_model.keras",
        "saved_models/ann_model.keras",
        "./saved_models/ann_model.keras",
        os.path.join(PROJECT_ROOT, "saved_models", "ann_model.keras"),
        os.path.join(os.getcwd(), "saved_models", "ann_model.keras"),
        os.path.join(os.path.dirname(os.getcwd()), "saved_models", "ann_model.keras"),
        # Add absolute paths relative to this file
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "ann_model.keras"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", "ann_model.keras"),
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    return None

def load_model():
    """Load model and scaler with error handling"""
    # Try to find model file
    model_path = find_model_file()
    
    if model_path is None:
        # Print debug information
        debug_info = {
            "Current working directory": os.getcwd(),
            "Project root": PROJECT_ROOT,
            "File location": os.path.abspath(__file__),
            "MODEL_PATH": MODEL_PATH,
            "Files in current directory": os.listdir('.') if os.path.exists('.') else 'N/A',
        }
        
        # Check for saved_models directory
        saved_models_paths = [
            "saved_models",
            "../saved_models",
            os.path.join(PROJECT_ROOT, "saved_models"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models"),
        ]
        
        for sp in saved_models_paths:
            if os.path.exists(sp):
                debug_info[f"Files in {sp}"] = os.listdir(sp)
        
        # Print debug info
        print("=" * 60)
        print("DEBUG INFO - Model not found")
        print("=" * 60)
        for key, value in debug_info.items():
            print(f"{key}: {value}")
        print("=" * 60)
        
        raise FileNotFoundError(
            f"Model file not found. Please ensure 'ann_model.keras' exists in the 'saved_models' directory.\n"
            f"Project root: {PROJECT_ROOT}\n"
            f"Expected path: {MODEL_PATH}\n"
            f"Current directory: {os.getcwd()}"
        )
    
    try:
        print(f"✅ Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Find scaler file
        scaler_path = None
        possible_scaler_paths = [
            model_path.replace("ann_model.keras", "scaler.joblib"),
            os.path.join(os.path.dirname(model_path), "scaler.joblib"),
            SCALER_PATH,
            "../saved_models/scaler.joblib",
            "saved_models/scaler.joblib",
            os.path.join(PROJECT_ROOT, "saved_models", "scaler.joblib"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "scaler.joblib"),
        ]
        
        for sp in possible_scaler_paths:
            if sp and os.path.exists(sp):
                scaler_path = sp
                break
        
        if scaler_path is None:
            raise FileNotFoundError(f"Scaler file not found. Searched in: {possible_scaler_paths}")
        
        print(f"✅ Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        print("✅ Model and scaler loaded successfully!")
        return model, scaler
    
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")
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
            df[col] = 0  # Add missing columns with default value
    
    # Convert to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill NaN with 0
        df[col] = df[col].fillna(0)
    
    # Select only the required columns in correct order
    df = df[numeric_cols]
    
    # Scale the data
    try:
        scaled = scaler.transform(df)
        return scaled
    except Exception as e:
        print(f"Error in scaling: {e}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame dtypes: {df.dtypes}")
        raise

# Cache the model loading for Streamlit
_model_cache = None
_scaler_cache = None

def get_cached_model():
    """Get cached model and scaler for Streamlit"""
    global _model_cache, _scaler_cache
    if _model_cache is None or _scaler_cache is None:
        _model_cache, _scaler_cache = load_model()
    return _model_cache, _scaler_cache

# Test the module
if __name__ == "__main__":
    try:
        model, scaler = load_model()
        print("Model and scaler loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Scaler type: {type(scaler)}")
    except Exception as e:
        print(f"Failed to load: {e}")