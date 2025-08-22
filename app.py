import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Website Traffic Forecasting", page_icon="📈", layout="wide")
st.title("📈 Website Traffic Forecasting")

uploaded = st.file_uploader("Upload CSV with columns: Date, Page.Loads", type=["csv"])
periods = st.slider("Forecast horizon (days)", 7, 90, 30, step=1)

if uploaded:
    df = pd.read_csv(uploaded)
    df["Date"] = pd.to_datetime(df["Date"])
    data = df[["Date","Page.Loads"]].rename(columns={"Date":"ds","Page.Loads":"y"})
    m = Prophet(daily_seasonality=True)
    m.fit(data)
    future = m.make_future_dataframe(periods=periods)
    fc = m.predict(future)

    st.subheader("Forecast")
    fig1 = m.plot(fc); st.pyplot(fig1)

    st.subheader("Forecast Components")
    fig2 = m.plot_components(fc); st.pyplot(fig2)
else:
    st.info("Upload a CSV to get started.")
