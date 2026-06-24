import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_visual(df):

    st.subheader("Visualisasi Dataset")

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x="Price",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    fig2, ax2 = plt.subplots()

    sns.boxplot(
        data=df,
        x="Company",
        y="Price",
        ax=ax2
    )

    plt.xticks(rotation=45)

    st.pyplot(fig2)