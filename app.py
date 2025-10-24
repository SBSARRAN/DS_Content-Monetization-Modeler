import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, and columns
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
X_train_columns = joblib.load("X_train_columns.pkl")

st.title("🎬 YouTube Ad Revenue Predictor")
st.write("Enter video and channel details below to estimate ad revenue.")

# --- Numeric Inputs ---
views = st.number_input("Views", min_value=0)
comments = st.number_input("Comments", min_value=0)
video_length_minutes = st.number_input("Video Length (minutes)", min_value=0.0)
subscribers = st.number_input("Subscribers", min_value=0)
watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0.0)

# Calculate engagement rate automatically
engagement_rate = (comments / views) if views > 0 else 0

# --- Categorical Inputs ---
category = st.selectbox("Category", ["Entertainment", "Gaming", "Lifestyle", "Music", "Tech"])
device = st.selectbox("Device", ["Mobile", "TV", "Tablet"])
country = st.selectbox("Country", ["CA", "DE", "IN", "UK", "US"])

# Predict Button
if st.button("Predict Ad Revenue"):
    # Build input dataframe
    input_dict = {
        'views': views,
        'comments': comments,
        'video_length_minutes': video_length_minutes,
        'subscribers': subscribers,
        'engagement_rate': engagement_rate,
        'watch_time_minutes': watch_time_minutes,
        f'category_{category}': 1,
        f'device_{device}': 1,
        f'country_{country}': 1
    }

    # Fill missing dummy columns with 0
    input_df = pd.DataFrame([input_dict])
    for col in X_train_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X_train_columns]

    # Scale numeric features
    num_cols = ['views', 'comments', 'video_length_minutes', 'subscribers', 'engagement_rate', 'watch_time_minutes']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Ad Revenue: **${prediction:,.2f} USD**")
