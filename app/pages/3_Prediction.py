import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Voice Prediction")

# Load models
classifier = joblib.load("../models/classifier_model.pkl")
cluster = joblib.load("../models/cluster_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Load dataset
df = pd.read_csv("../data/vocal_gender_features.csv")

features = df.drop("label", axis=1).columns

st.subheader("Enter Voice Feature Values")

# store inputs
inputs = []

for feature in features:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

# Predict button
if st.button("Predict"):

    input_df = pd.DataFrame([inputs], columns=features)

    scaled = scaler.transform(input_df)

    gender = classifier.predict(scaled)[0]

    cluster_id = cluster.predict(scaled)[0]

    st.subheader("Prediction Results")

    st.write("Cluster:", cluster_id)

    if gender == 1:
        st.success("Male Voice")
    else:
        st.success("Female Voice")

# Load dataset sample
st.subheader("Test Using Dataset Row")

if st.button("Load Sample Data"):

    sample = df.sample(1)

    st.write("Sample Row From Dataset")

    st.dataframe(sample)

    actual = sample["label"].values[0]

    X = sample.drop("label", axis=1)

    scaled = scaler.transform(X)

    pred = classifier.predict(scaled)[0]

    st.subheader("Model Verification")

    st.write("Actual Label:", "Male" if actual == 1 else "Female")

    st.write("Predicted Label:", "Male" if pred == 1 else "Female")