import streamlit as st
import pandas as pd
import os
import sys

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
try:
    from pages.prediction import predict_price
    from pages.visualisasi import show_visual
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Laptop Price Prediction App")
st.write("Deploy TensorFlow ANN + Streamlit")

# ============================================
# FIND AND LOAD DATA - FIXED FOR SEMICOLON CSV
# ============================================

def find_data_file():
    """Find the laptop data CSV file"""
    possible_paths = [
        "../data/laptopData.csv",
        "data/laptopData.csv",
        "./data/laptopData.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "laptopData.csv"),
        os.path.join(os.getcwd(), "data", "laptopData.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_laptop_data():
    """Load laptop data with proper parsing"""
    data_path = find_data_file()
    
    if data_path is None:
        st.warning("⚠️ Data file not found.")
        return pd.DataFrame()
    
    try:
        # Try reading with semicolon separator
        df = pd.read_csv(data_path, sep=';', encoding='utf-8')
        
        # Remove rows that are all NaN or empty
        df = df.dropna(how='all')
        
        # Remove rows where the first column is empty (those rows with just semicolons)
        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.strip().isin(['', 'nan', 'None'])]
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        # Try to find and clean Price column
        for col in df.columns:
            if col.lower().strip() == 'price':
                # Clean price: remove dots and convert to numeric
                df[col] = df[col].astype(str).str.replace('.', '', regex=False)
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                break
        
        # Display info
        st.sidebar.success(f"✅ Data loaded: {len(df)} rows")
        
        # Show columns in debug
        with st.sidebar.expander("📊 Data Info"):
            st.write(f"Columns: {df.columns.tolist()}")
            st.write(f"Shape: {df.shape}")
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_laptop_data()

# Sidebar menu
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Visualisasi", "Prediksi"]
)

if menu == "Visualisasi":
    if not df.empty:
        st.header("📊 Visualisasi Data")
        
        # Show first few rows
        with st.expander("Lihat Data"):
            st.dataframe(df.head())
        
        # Show visualizations
        try:
            show_visual(df)
        except Exception as e:
            st.error(f"Error showing visualizations: {e}")
            st.exception(e)  # This will show the full traceback
    else:
        st.warning("Data tidak tersedia untuk visualisasi")
        st.info("Pastikan file 'laptopData.csv' ada di folder 'data' dengan format yang benar.")

else:
    st.header("🔮 Prediksi Harga Laptop")
    
    # Check if model files exist
    model_paths = [
        "../saved_models/ann_model.keras",
        "saved_models/ann_model.keras",
        "./saved_models/ann_model.keras",
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            model_found = True
            break
    
    if not model_found:
        st.error("❌ Model files not found. Please ensure 'ann_model.keras' and 'scaler.joblib' are in the 'saved_models' directory.")
        
        with st.expander("Debug Info"):
            st.write("Current directory:", os.getcwd())
            st.write("Files in current directory:", os.listdir('.') if os.path.exists('.') else 'N/A')
            if os.path.exists('saved_models'):
                st.write("Files in saved_models:", os.listdir('saved_models'))
            if os.path.exists('../saved_models'):
                st.write("Files in ../saved_models:", os.listdir('../saved_models'))
        st.stop()
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            inches = st.number_input(
                "Ukuran Layar (Inches)",
                min_value=10.0,
                max_value=20.0,
                value=15.6,
                step=0.1,
                help="Ukuran layar laptop dalam inch"
            )
            
            ram = st.selectbox(
                "RAM (GB)",
                options=[4, 8, 16, 32, 64],
                index=1,
                help="Kapasitas RAM dalam GB"
            )
            
            weight = st.number_input(
                "Berat (kg)",
                min_value=0.5,
                max_value=5.0,
                value=2.5,
                step=0.1,
                help="Berat laptop dalam kilogram"
            )
            
            touchscreen = st.selectbox(
                "Touchscreen",
                options=[0, 1],
                format_func=lambda x: "Ya" if x == 1 else "Tidak",
                help="Apakah laptop memiliki layar sentuh?"
            )
            
            ips = st.selectbox(
                "IPS Panel",
                options=[0, 1],
                format_func=lambda x: "Ya" if x == 1 else "Tidak",
                help="Apakah laptop menggunakan panel IPS?"
            )
        
        with col2:
            x_res = st.number_input(
                "Resolusi X",
                min_value=800,
                max_value=3840,
                value=1920,
                step=100,
                help="Resolusi horizontal dalam pixel"
            )
            
            y_res = st.number_input(
                "Resolusi Y",
                min_value=600,
                max_value=2160,
                value=1080,
                step=100,
                help="Resolusi vertical dalam pixel"
            )
            
            hdd = st.number_input(
                "HDD (GB)",
                min_value=0,
                max_value=2000,
                value=0,
                step=100,
                help="Kapasitas HDD dalam GB"
            )
            
            ssd = st.number_input(
                "SSD (GB)",
                min_value=0,
                max_value=2000,
                value=512,
                step=100,
                help="Kapasitas SSD dalam GB"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Prediksi Harga", use_container_width=True)
    
    if submitted:
        try:
            # Create input DataFrame
            input_df = pd.DataFrame([{
                "Inches": inches,
                "Ram": ram,
                "Weight": weight,
                "Touchscreen": touchscreen,
                "IPS Panel": ips,
                "X_res": x_res,
                "Y_res": y_res,
                "HDD": hdd,
                "SSD": ssd
            }])
            
            # Make prediction
            with st.spinner("Memprediksi harga..."):
                hasil = predict_price(input_df)
            
            # Display result
            st.success(f"💰 Estimasi Harga Laptop: **Rp {hasil:,.0f}**")
            
            # Show confidence (if available)
            st.info(f"💡 Berdasarkan spesifikasi yang Anda masukkan")
            
        except Exception as e:
            st.error(f"❌ Error dalam prediksi: {e}")
            st.error("Pastikan model file tersedia di folder 'saved_models'")
            
            # Debug info
            with st.expander("Debug Info"):
                st.write("Current directory:", os.getcwd())
                st.write("Files in current directory:", os.listdir('.'))
                if os.path.exists('../saved_models'):
                    st.write("Files in saved_models:", os.listdir('../saved_models'))