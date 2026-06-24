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
# FILE FINDING FUNCTIONS - WORKS IN CLOUD
# ============================================

def get_project_root():
    """Get the project root directory"""
    # In Streamlit Cloud, the app runs from the root of the repo
    # Check if we're in a cloud environment
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

def find_data_file():
    """Find the laptop data CSV file"""
    project_root = get_project_root()
    
    possible_paths = [
        # Relative paths
        "../data/laptopData.csv",
        "data/laptopData.csv",
        "./data/laptopData.csv",
        # Absolute paths from project root
        os.path.join(project_root, "data", "laptopData.csv"),
        os.path.join(project_root, "laptopData.csv"),
        # Current directory
        os.path.join(os.getcwd(), "data", "laptopData.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "laptopData.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "laptopData.csv"),
    ]
    
    # Debug: Print all paths being checked
    st.sidebar.write("🔍 Searching for data file...")
    
    for path in possible_paths:
        if path and os.path.exists(path):
            st.sidebar.success(f"✅ Found data at: {path}")
            return path
    
    # If not found, list files for debugging
    st.sidebar.warning("⚠️ Data file not found. Checking directory contents...")
    try:
        files = os.listdir('.')
        st.sidebar.write(f"Files in current directory: {files}")
        
        if os.path.exists('data'):
            st.sidebar.write(f"Files in data/: {os.listdir('data')}")
        if os.path.exists('../data'):
            st.sidebar.write(f"Files in ../data/: {os.listdir('../data')}")
    except Exception as e:
        st.sidebar.write(f"Error listing files: {e}")
    
    return None

def find_model_files():
    """Find model and scaler files"""
    project_root = get_project_root()
    
    model_path = None
    scaler_path = None
    
    possible_model_paths = [
        os.path.join(project_root, "saved_models", "ann_model.keras"),
        os.path.join(os.getcwd(), "saved_models", "ann_model.keras"),
        "../saved_models/ann_model.keras",
        "saved_models/ann_model.keras",
        "./saved_models/ann_model.keras",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", "ann_model.keras"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "ann_model.keras"),
    ]
    
    possible_scaler_paths = [
        os.path.join(project_root, "saved_models", "scaler.joblib"),
        os.path.join(os.getcwd(), "saved_models", "scaler.joblib"),
        "../saved_models/scaler.joblib",
        "saved_models/scaler.joblib",
        "./saved_models/scaler.joblib",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", "scaler.joblib"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "scaler.joblib"),
    ]
    
    for path in possible_model_paths:
        if path and os.path.exists(path):
            model_path = path
            break
    
    for path in possible_scaler_paths:
        if path and os.path.exists(path):
            scaler_path = path
            break
    
    return model_path, scaler_path

# ============================================
# LOAD DATA
# ============================================

# Find and load data
data_path = find_data_file()
if data_path:
    try:
        df = pd.read_csv(data_path)
        st.sidebar.success(f"✅ Data loaded: {len(df)} rows")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()
else:
    st.warning("⚠️ Data file not found. Please ensure 'laptopData.csv' exists in the 'data' directory.")
    st.info(f"Current directory: {os.getcwd()}")
    st.info(f"Project root: {get_project_root()}")
    df = pd.DataFrame()

# ============================================
# SIDEBAR MENU
# ============================================

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
    else:
        st.warning("Data tidak tersedia untuk visualisasi")

else:
    st.header("🔮 Prediksi Harga Laptop")
    
    # Check if model files exist
    model_path, scaler_path = find_model_files()
    
    if not model_path or not scaler_path:
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
                st.write("Files in current directory:", os.listdir('.') if os.path.exists('.') else 'N/A')
                if os.path.exists('../saved_models'):
                    st.write("Files in saved_models:", os.listdir('../saved_models'))
                if os.path.exists('saved_models'):
                    st.write("Files in saved_models:", os.listdir('saved_models'))