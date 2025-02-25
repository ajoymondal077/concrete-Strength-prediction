import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError # type: ignore


# Explicitly specify the loss function
model = tf.keras.models.load_model('concrete_strength.h5', custom_objects={'mse': MeanSquaredError()})
scaler = joblib.load('scaler.pkl')

st.title("Concrete Strength prediction")
st.write("Enter the input parameters to predict the concrete strength (MPa).")
cement = st.number_input("Cement (kg/m³)", min_value=0.0, format="%.2f")
blast_furnace_slag = st.number_input("Blast furnace slag (kg/m³)", min_value=0.0, format="%.2f")
fly_ash = st.number_input("Fly ash (kg/m³)", min_value=0.0, format="%.2f")
water = st.number_input("Water (kg/m³)", min_value=0.0, format="%.2f")
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, format="%.2f")
coarse_aggregate = st.number_input("coarse_aggregate (kg/m³)", min_value=0.0, format="%.2f")
fine_aggregate = st.number_input("fine_aggregate (kg/m³)", min_value=0.0, format="%.2f")
age = st.number_input("Age (days)", min_value=0, format="%d")

if st.button("Predict Strength"):
    user_data = np.array([[cement, blast_furnace_slag, fly_ash, water,
                           superplasticizer, coarse_aggregate, fine_aggregate, age]])

    # Scale the user input
    user_data_df = pd.DataFrame(user_data, columns=["Cement", "BlastFurnaceSlag", "FlyAsh", "Water","Superplasticizer", "CoarseAggregate", "FineAggregate", "Age"])
    user_data_scaled = scaler.transform(user_data_df)
    prediction = model.predict(user_data_scaled)
    st.success(f"Predicted Concrete Strength: **{prediction[0][0]:.2f} MPa**")
    