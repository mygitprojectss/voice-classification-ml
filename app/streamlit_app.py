import streamlit as st

st.set_page_config(
    page_title="Human Voice ML System",
    layout="wide"
)

st.title("🎤 Human Voice Classification System")

st.markdown("""
### Project Overview

This application performs **Human Voice Analysis using Machine Learning**.

The system includes:

• Voice Feature Exploration  
• Voice Clustering  
• Gender Classification  
• Interactive Prediction  

### Machine Learning Pipeline

1️⃣ Data Preprocessing  
2️⃣ Feature Scaling  
3️⃣ Clustering (KMeans)  
4️⃣ Classification (Random Forest)  
5️⃣ Model Evaluation  
6️⃣ Streamlit Deployment  

Use the **sidebar to navigate through the application**.
""")

st.info("Navigate using the sidebar to explore the project.")