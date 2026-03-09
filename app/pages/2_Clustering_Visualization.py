import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

st.title("🔍 Clustering Visualization")

df = pd.read_csv("../data/vocal_gender_features.csv")

cluster_model = joblib.load("../models/cluster_model.pkl")

X = df.drop("label", axis=1)

clusters = cluster_model.predict(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

fig, ax = plt.subplots()

scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters)

ax.set_title("Voice Clusters (PCA Projection)")

st.pyplot(fig)