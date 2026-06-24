import numpy as np
from essential import load_model, preprocess_input


model, scaler = load_model()


def predict_price(input_df):

    data = preprocess_input(input_df, scaler)

    pred = model.predict(data)

    harga = np.exp(pred[0][0])

    return round(harga, 2)