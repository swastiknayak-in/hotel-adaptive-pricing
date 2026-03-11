# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from model_training import train_and_save, load_data, preprocess_features
import plotly.express as px

MODEL_PATH = 'models/pricing_model.pkl'
DATA_PATH = 'data/hotel_bookings.csv'

st.set_page_config(page_title='Hotel Adaptive Pricing', layout='wide')

st.title('Customer Behavior Analysis Based Adaptive Pricing System for Hotel Management')
st.markdown('''
This app trains a pricing model (automatically on first run) and shows interactive dashboards. Enter booking details to get a recommended ADR (room price).
''')

# Ensure model exists (auto-train if missing)
if not os.path.exists(MODEL_PATH):
    with st.spinner('Training pricing model (this may take a minute)...'):
        model_file = train_and_save(DATA_PATH, model_dir='models')
    st.success('Model trained and saved.')

# Load model
model = joblib.load(MODEL_PATH)

# Load data for dashboards
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f'Could not load dataset at {DATA_PATH}: {e}')
    st.stop()

# Preprocess for visualizations
# Clean month names to numeric for plotting month trend
def month_to_num(m):
    months = {
        'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
        'July':7,'August':8,'September':9,'October':10,'November':11,'December':12
    }
    return months.get(m, None)

if 'arrival_date_month' in df.columns:
    df['arrival_month_num'] = df['arrival_date_month'].map(lambda x: month_to_num(x))

# Layout: two columns for dashboards and prediction
left, right = st.columns([2,1])

with left:
    st.subheader('Customer Behavior Dashboard')
    # 1. Customer loyalty distribution
    st.markdown('**Customer loyalty (previous successful bookings)**')
    fig1 = px.histogram(df, x='previous_bookings_not_canceled', nbins=20, title='Previous successful bookings distribution')
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Lead time distribution
    st.markdown('**Lead time distribution (days between booking and arrival)**')
    fig2 = px.histogram(df, x='lead_time', nbins=50, title='Lead time distribution')
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Customer type distribution
    st.markdown('**Customer type distribution**')
    if 'customer_type' in df.columns:
        fig3 = px.pie(df, names='customer_type', title='Customer type share')
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Room type preference
    st.markdown('**Room type preference (reserved)**')
    if 'reserved_room_type' in df.columns:
        top_rooms = df['reserved_room_type'].value_counts().reset_index()
        top_rooms.columns = ['room','count']
        fig4 = px.bar(top_rooms, x='room', y='count', title='Reserved room types')
     