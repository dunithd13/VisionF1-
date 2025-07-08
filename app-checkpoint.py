import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Streamlit page config
st.set_page_config(page_title="VisionF1", layout="wide")

# Custom CSS for white background and black text
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
        }
        .css-18e3th9 {
            background-color: white;
        }
        .css-1d391kg {
            color: black;
        }
        .stApp {
            background-color: white;
        }
        .stButton>button {
            background-color: #f63366;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Load ML models (make sure they're saved in your project directory)
position_model = joblib.load('position_model.pkl')
points_model = joblib.load('points_model.pkl')

# Load feature data
feature_data = pd.read_csv('future_features.csv')  # Match the features expected by model

# Display logo centered at top
st.image('VisionF1/logo.png', width=200)
st.markdown("<h1 style='text-align: center;'>🏁 VisionF1 – 2025 Race Predictor</h1>", unsafe_allow_html=True)

# Driver and Race selections
drivers = feature_data['driver'].unique()
races = feature_data['race'].unique()

col1, col2 = st.columns(2)
with col1:
    selected_driver = st.selectbox("Select Driver", sorted(drivers))
with col2:
    selected_race = st.selectbox("Select Race", sorted(races))

# Predict button
if st.button("Predict Position & Points"):
    # Filter the data for selected driver and race
    input_row = feature_data[
        (feature_data['driver'] == selected_driver) &
        (feature_data['race'] == selected_race)
    ]

    if not input_row.empty:
        # Predict
        predicted_position = position_model.predict(input_row)[0]
        points_prob = points_model.predict_proba(input_row)[0][1]  # Prob of class 1

        # Display results
        st.markdown(f"### Prediction for **{selected_driver}** at **{selected_race}**")
        colA, colB = st.columns(2)
        with colA:
            st.metric("🏁 Predicted Finish Position", f"{round(predicted_position)}")
        with colB:
            st.metric("💯 Probability of Scoring Points", f"{points_prob:.1%}")

    else:
        st.warning("Prediction data not found for this selection. Please check your dataset.")

# Display driver group image at the bottom
st.markdown("---")
st.image("VisionF1/F1image.jpg", use_column_width=True)
