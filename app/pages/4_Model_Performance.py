import streamlit as st

st.title("📈 Model Performance")

st.markdown("""
### Classification Model

Algorithm: **Random Forest**

Accuracy Achieved: **99%**

Metrics Used:

• Accuracy  
• Precision  
• Recall  
• F1 Score  

---

### Clustering Model

Algorithm: **KMeans**

Metric Used:

• Silhouette Score

Result:

Silhouette Score ≈ **0.17**

---

### Observations

Classification performs extremely well while clustering provides moderate separation between voice groups.
""")