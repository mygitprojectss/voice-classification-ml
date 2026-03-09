import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models
classifier = joblib.load("../models/classifier_model.pkl")
cluster = joblib.load("../models/cluster_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

st.title("Human Voice Classification System")

st.write("Enter Voice Feature Values")

# Load dataset to get feature names
df = pd.read_csv("../data/vocal_gender_features.csv")

feature_names = df.drop("label", axis=1).columns

inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)

if st.button("Predict"):

    arr = pd.DataFrame([inputs], columns=feature_names)

    scaled = scaler.transform(arr)

    gender = classifier.predict(scaled)[0]

    cluster_id = cluster.predict(scaled)[0]

    st.write("Cluster:", cluster_id)

    if gender == 1:
        st.success("Prediction: Male")
    else:
        st.success("Prediction: Female")