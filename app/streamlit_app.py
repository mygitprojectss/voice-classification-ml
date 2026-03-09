import streamlit as st

st.set_page_config(
    page_title="Voice Gender Classification",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Voice Gender Classification using Machine Learning")
st.divider()
col1, col2, col3 = st.columns(3)

col1.metric("Dataset Size", "3,168 Samples")
col2.metric("Features Used", "20+ Audio Features")
col3.metric("Model Accuracy", "95%")

st.markdown("""
## 📌 Project Overview

This application predicts **Male or Female voice** using **Machine Learning models** trained on extracted audio features.

The system analyzes acoustic features such as:

- Pitch
- Spectral Centroid
- Spectral Bandwidth
- MFCC Features
- Energy
- Zero Crossing Rate

Using these features, the model predicts whether the voice belongs to **Male or Female**.

---

## ⚙️ Technologies Used

- Python
- Scikit-Learn
- Librosa
- Pandas
- NumPy
- Streamlit

---

## 🧠 Machine Learning Models

The project uses:

• Classification Model → Gender Prediction  
• Clustering Model → Voice Pattern Grouping  
• StandardScaler → Feature Scaling  

---

## 📊 Application Pages

Use the **sidebar navigation** to explore the modules:

### 📈 EDA Visualization
Explore dataset distributions and feature relationships.

### 🧩 Clustering Visualization
Visualize clustering of voice patterns.

### 🔮 Prediction
Test the ML model using:
- Manual feature input
- Dataset testing
- Upload `.wav` audio

### 📉 Model Performance
View accuracy and evaluation metrics.

---

## 🚀 Workflow

Audio Voice  
↓  
Feature Extraction (Librosa)  
↓  
Feature Scaling  
↓  
Machine Learning Model  
↓  
Gender Prediction  

---

### 👉 Use the **left sidebar** to navigate through the project pages.
""")

st.divider()

st.markdown(
"""
### 👨‍💻 Developed By

**AI Voice Classification Project**

Built using **Python, Scikit-Learn, Librosa, and Streamlit**
"""
)