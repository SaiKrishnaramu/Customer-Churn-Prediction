"""
📡 Telco Customer Churn Prediction — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG & SETUP
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHURN_COLOR  = "#EF553B"
RETAIN_COLOR = "#00CC96"

# Features exactly as expected by the notebook's model
FEATURE_COLS = [
    "State", "Account length", "Area code", "International plan", 
    "Voice mail plan", "Number vmail messages", "Total day minutes", 
    "Total day calls", "Total day charge", "Total eve minutes", 
    "Total eve calls", "Total eve charge", "Total night minutes", 
    "Total night calls", "Total night charge", "Total intl minutes", 
    "Total intl calls", "Total intl charge", "Customer service calls"
]

# Load the model saved from the Jupyter Notebook
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# ─────────────────────────────────────────────────────────────
# UI: SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("👤 Customer Profile")

if not model:
    st.sidebar.error("⚠️ `model.pkl` not found. Please run the notebook first to generate the model.")
    st.stop()

# Categorical & Boolean Inputs
state = st.sidebar.selectbox("State (Code)", options=[
    'KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT',
    'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL',
    'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD',
    'NC', 'WA', 'NM', 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND'
])
area_code = st.sidebar.selectbox("Area Code", options=[415, 408, 510])
intl_plan = st.sidebar.radio("International Plan", ["No", "Yes"])
vmail_plan = st.sidebar.radio("Voice Mail Plan", ["No", "Yes"])

# Numeric Inputs
st.sidebar.markdown("---")
st.sidebar.header("📊 Usage Metrics")

col1, col2 = st.sidebar.columns(2)
account_length = col1.number_input("Account Length (days)", min_value=1, value=100)
cust_service_calls = col2.number_input("Service Calls", min_value=0, value=1)

vmail_msgs = st.sidebar.number_input("Voicemail Messages", min_value=0, value=0 if vmail_plan == "No" else 20)

st.sidebar.markdown("**Day Usage**")
day_mins = st.sidebar.number_input("Day Minutes", min_value=0.0, value=150.0)
day_calls = st.sidebar.number_input("Day Calls", min_value=0, value=100)
day_charge = day_mins * 0.17  # Approximate calculation based on dataset

st.sidebar.markdown("**Evening Usage**")
eve_mins = st.sidebar.number_input("Evening Minutes", min_value=0.0, value=200.0)
eve_calls = st.sidebar.number_input("Evening Calls", min_value=0, value=100)
eve_charge = eve_mins * 0.085

st.sidebar.markdown("**Night Usage**")
night_mins = st.sidebar.number_input("Night Minutes", min_value=0.0, value=200.0)
night_calls = st.sidebar.number_input("Night Calls", min_value=0, value=100)
night_charge = night_mins * 0.045

st.sidebar.markdown("**International Usage**")
intl_mins = st.sidebar.number_input("Intl Minutes", min_value=0.0, value=10.0)
intl_calls = st.sidebar.number_input("Intl Calls", min_value=0, value=3)
intl_charge = intl_mins * 0.27

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
# Note: Label encoding logic should match exactly how it was fit in the notebook. 
# For HistGradientBoostingClassifier, it often handles categorical natively if configured, 
# or requires basic encoding. Assuming ordinal/basic encoding for State and binary for plans.

state_mapping = {s: i for i, s in enumerate([
    'KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT',
    'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL',
    'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD',
    'NC', 'WA', 'NM', 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND'
])}

input_data = pd.DataFrame([{
    "State": state_mapping.get(state, 0),
    "Account length": account_length,
    "Area code": area_code,
    "International plan": 1 if intl_plan == "Yes" else 0,
    "Voice mail plan": 1 if vmail_plan == "Yes" else 0,
    "Number vmail messages": vmail_msgs,
    "Total day minutes": day_mins,
    "Total day calls": day_calls,
    "Total day charge": day_charge,
    "Total eve minutes": eve_mins,
    "Total eve calls": eve_calls,
    "Total eve charge": eve_charge,
    "Total night minutes": night_mins,
    "Total night calls": night_calls,
    "Total night charge": night_charge,
    "Total intl minutes": intl_mins,
    "Total intl calls": intl_calls,
    "Total intl charge": intl_charge,
    "Customer service calls": cust_service_calls
}])

# ─────────────────────────────────────────────────────────────
# PREDICTION & MAIN UI
# ─────────────────────────────────────────────────────────────
st.title("📡 Telco Churn Predictor")
st.markdown("Quantify revenue at risk and enable targeted retention strategies.")

# Run Inference
prob_churn = model.predict_proba(input_data)[0][1]
is_churn = model.predict(input_data)[0]

# Risk Tiers aligned with Business Impact
if prob_churn >= 0.80:
    tier = "🔴 Critical Risk"
    alert_color = "#f8d7da"
elif prob_churn >= 0.50:
    tier = "🟠 Warning"
    alert_color = "#fff3cd"
else:
    tier = "🟢 Safe / Low Risk"
    alert_color = "#d4edda"

col_a, col_b = st.columns([1, 1.5])

with col_a:
    st.markdown("### Risk Assessment")
    st.markdown(f"**Customer Risk Tier:** {tier}")
    
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_churn * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": CHURN_COLOR if prob_churn >= 0.5 else RETAIN_COLOR},
            "steps": [
                {"range": [0, 50], "color": "#d4edda"},
                {"range": [50, 80], "color": "#fff3cd"},
                {"range": [80, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": prob_churn * 100,
            },
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("### 📋 Business Intervention Strategy")
    st.info("Based on the generated model profile, here is the recommended retention action:")
    
    # Logic tied to the notebook's expected business outcomes
    if prob_churn >= 0.50:
        if cust_service_calls >= 3:
            st.error("🚨 **High Service Calls Detected:** Trigger immediate escalation team intervention before the customer decides to leave.")
        elif intl_plan == "Yes":
            st.warning("🌐 **International Plan User:** Review plan pricing vs competitors. Offer a specialized retention discount on international rates.")
        elif account_length < 30:
            st.warning("🆕 **New Account (<30 Days):** High early churn risk. Schedule a dedicated onboarding call immediately.")
        else:
            st.error("⚠️ **General High Risk:** Call this customer personally. Offer a plan upgrade or loyalty discount to protect recurring revenue.")
    else:
        st.success("✅ **Stable Customer:** Continue standard engagement. Retaining high-value, low-risk customers adds disproportionate lifetime value.")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:13px'>"
    "📡 Model: HistGradientBoostingClassifier | Dataset: BigML Telco Churn"
    "</p>", unsafe_allow_html=True
)
