import streamlit as st
import pandas as pd
from pages.prediction import predict_price
from pages.visualisasi import show_visual


st.set_page_config(
    page_title="Laptop Price Prediction",
    layout="wide"
)

st.title("💻 Laptop Price Prediction App")
st.write("Deploy TensorFlow ANN + Streamlit")


df = pd.read_csv("../data/laptopData.csv")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Visualisasi",
        "Prediksi"
    ]
)

if menu == "Visualisasi":

    st.dataframe(df.head())

    show_visual(df)

else:

    st.header("Input Data Laptop")

    inches = st.number_input(
        "Ukuran Layar",
        10.0,
        20.0
    )

    ram = st.selectbox(
        "RAM",
        [4, 8, 16, 32]
    )

    weight = st.number_input(
        "Berat",
        0.5,
        5.0
    )

    touchscreen = st.selectbox(
        "Touchscreen",
        [0, 1]
    )

    ips = st.selectbox(
        "IPS Panel",
        [0, 1]
    )

    x = st.number_input(
        "Resolusi X",
        value=1920
    )

    y = st.number_input(
        "Resolusi Y",
        value=1080
    )

    hdd = st.number_input(
        "HDD",
        value=0
    )

    ssd = st.number_input(
        "SSD",
        value=512
    )

    if st.button("Prediksi"):

        input_df = pd.DataFrame([{

            "Inches": inches,
            "Ram": ram,
            "Weight": weight,
            "Touchscreen": touchscreen,
            "IPS Panel": ips,
            "X_res": x,
            "Y_res": y,
            "HDD": hdd,
            "SSD": ssd
        }])

        hasil = predict_price(input_df)

        st.success(
            f"Estimasi Harga Laptop: ₹ {hasil:,.0f}"
        )