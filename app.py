import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

# Page Config (fixed)
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌫️",
    layout="wide"
)

# --- Custom CSS for styling ---
st.markdown("""
<style>
    .main {
        background-color: #f3f7fb;
    }
    .title {
        font-size: 42px;
        color: #2C3E50;
        font-weight: 800;
        text-align: center;
        padding-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #5D6D7E;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #21618C;
        color: #ECF0F1;
    }
    .feature-box {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgb(0 0 0 / 8%);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='title'>🌫️ Air Quality Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter the environmental parameters below to predict the Air Quality Index (AQI) category.</p>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("ℹ️ App Information")
st.sidebar.write("""
### About the Model
This application uses an **Ensemble Voting Classifier**  
combined with:

- 🌿 LightGBM  
- ⚡ XGBoost  
- 🌲 Random Forest  

to predict the **AQI Bucket** from air quality parameters.

### AQI Categories:
- 0 → Good  
- 1 → Moderate  
- 2 → Unhealthy  
- 3 → Very Unhealthy  
- 4 → Hazardous  
""")

# --- Feature Inputs ---
st.subheader("📥 Enter Feature Values")

col1, col2 = st.columns(2)

feature_values = []
feature_names = model.feature_name_

for index, f in enumerate(feature_names):
    with col1 if index % 2 == 0 else col2:
        st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
        val = st.number_input(f"🔹 {f}", value=0.0, step=0.1)
        st.markdown("</div>", unsafe_allow_html=True)
        feature_values.append(val)

# --- Prediction ---
if st.button("🔍 Predict AQI Category"):
    pred = model.predict([feature_values])[0]
    st.success(f"🌟 **Predicted AQI Bucket:** `{pred}`")
    st.balloons()


#////////////////////////////////////////////////////////////////////////////
