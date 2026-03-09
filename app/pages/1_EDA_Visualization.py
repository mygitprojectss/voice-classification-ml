import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Exploratory Data Analysis")

df = pd.read_csv("../data/vocal_gender_features.csv")

st.write("Dataset Preview")

st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

feature = st.selectbox("Select Feature", df.columns[:-1])

fig, ax = plt.subplots()

sns.histplot(df[feature], kde=True, ax=ax)

st.pyplot(fig)

st.write("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(10,6))

sns.heatmap(df.corr(), cmap="coolwarm")

st.pyplot(fig2)