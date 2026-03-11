# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from model_training import train_and_save, load_data
import plotly.express as px

MODEL_PATH = "models/pricing_model.pkl"
DATA_PATH = "data/hotel_bookings.csv"

st.set_page_config(page_title="Hotel Adaptive Pricing", layout="wide")

st.title("Customer Behavior Analysis Based Adaptive Pricing System for Hotel Management")

st.markdown(
"""
This app trains a pricing model automatically and shows customer behavior analytics.
Users can enter booking information to get a recommended room price.
"""
)

# -------------------------------
# Train model automatically
# -------------------------------
if not os.path.exists(MODEL_PATH):

    with st.spinner("Training pricing model..."):

        train_and_save(DATA_PATH, model_dir="models")

    st.success("Model trained successfully")


# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load(MODEL_PATH)

# -------------------------------
# Load dataset
# -------------------------------
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


# -------------------------------
# Month Conversion
# -------------------------------
def month_to_num(month):

    months = {
        "January":1,"February":2,"March":3,"April":4,
        "May":5,"June":6,"July":7,"August":8,
        "September":9,"October":10,"November":11,"December":12
    }

    return months.get(month, None)


df["arrival_month_num"] = df["arrival_date_month"].map(month_to_num)


# -------------------------------
# Layout
# -------------------------------
left, right = st.columns([2,1])


# ===============================
# DASHBOARD SECTION
# ===============================
with left:

    st.subheader("Customer Behavior Dashboard")

    # Customer loyalty
    fig1 = px.histogram(
        df,
        x="previous_bookings_not_canceled",
        nbins=20,
        title="Customer Loyalty Distribution"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Lead time distribution
    fig2 = px.histogram(
        df,
        x="lead_time",
        nbins=50,
        title="Lead Time Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Customer type
    fig3 = px.pie(
        df,
        names="customer_type",
        title="Customer Type Distribution"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Room type preference
    room_counts = df["reserved_room_type"].value_counts().reset_index()
    room_counts.columns = ["room", "count"]

    fig4 = px.bar(
        room_counts,
        x="room",
        y="count",
        title="Reserved Room Type Distribution"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # Monthly price trend
    monthly_price = df.groupby("arrival_month_num")["adr"].mean().reset_index()

    fig5 = px.line(
        monthly_price,
        x="arrival_month_num",
        y="adr",
        title="Average Price by Month"
    )

    st.plotly_chart(fig5, use_container_width=True)


# ===============================
# PRICE PREDICTION SECTION
# ===============================
with right:

    st.subheader("Booking Information")

    hotel = st.selectbox(
        "Hotel Type",
        ["Resort Hotel","City Hotel"]
    )

    lead_time = st.slider(
        "Lead Time (Days)",
        0,
        365,
        30
    )

    month = st.selectbox(
        "Arrival Month",
        [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
        ]
    )

    customer_type = st.selectbox(
        "Customer Type",
        ["Transient","Contract","Transient-Party","Group"]
    )

    room_type = st.selectbox(
        "Room Type",
        ["A","B","C","D","E","F","G"]
    )

    previous_bookings = st.number_input(
        "Previous Successful Bookings",
        0,
        50,
        0
    )

    arrival_month = month_to_num(month)

    # Predict button
    if st.button("Predict Price"):

        input_df = pd.DataFrame({

            "hotel":[hotel],
            "lead_time":[lead_time],
            "arrival_month":[arrival_month],
            "reserved_room_type":[room_type],
            "customer_type":[customer_type],
            "previous_bookings_not_canceled":[previous_bookings]

        })

        prediction = model.predict(input_df)[0]

        st.success(f"Recommended Room Price: ₹ {round(prediction,2)}")
