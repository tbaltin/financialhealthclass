import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Financial Health Predictor",
    layout="wide"
)

st.title("Financial Health Classification Predictor")

st.write(
    "Enter a company's financial data below. "
    "The system predicts its financial health class as A, B, C, or D."
)

model = joblib.load("xgboost_model.pkl")

st.subheader("Company Financial Inputs")

col1, col2 = st.columns(2)

with col1:
    net_profit = st.text_input("Net Profit / Loss (€)", "50000")
    total_debt = st.text_input("Total Debt (€)", "400000")
    current_assets = st.text_input("Current Assets (€)", "300000")

with col2:
    short_term_debt = st.text_input("Short-term Debt (€)", "150000")

    ateco_sector = st.selectbox(
        "ATECO Sector",
        ["Construction", "IT Services", "Manufacturing", "Retail", "Other"]
    )

    legal_form = st.selectbox(
        "Legal Form",
        ["SRL", "SPA", "Individual Company", "Other"]
    )

    region = st.selectbox(
        "Region",
        ["Lazio", "Lombardia", "Campania", "Sicilia", "Other"]
    )

def to_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0

def signed_log(x):
    return np.sign(x) * np.log1p(abs(x))

net_profit = to_float(net_profit)
total_debt = to_float(total_debt)
current_assets = to_float(current_assets)
short_term_debt = to_float(short_term_debt)

log_net_profit = signed_log(net_profit)
log_total_debt = signed_log(total_debt)
working_capital = current_assets - short_term_debt

delta_profit = 0
delta_debt = 0

ateco_map = {
    "Construction": 0,
    "IT Services": 1,
    "Manufacturing": 2,
    "Retail": 3,
    "Other": 4
}

legal_form_map = {
    "SRL": 0,
    "SPA": 1,
    "Individual Company": 2,
    "Other": 3
}

region_map = {
    "Lazio": 0,
    "Lombardia": 1,
    "Campania": 2,
    "Sicilia": 3,
    "Other": 4
}

input_df = pd.DataFrame({
    "log_net_profit": [log_net_profit],
    "log_total_debt": [log_total_debt],
    "working_capital": [working_capital],
    "delta_profit": [delta_profit],
    "delta_debt": [delta_debt],
    "ateco_sector": [ateco_map[ateco_sector]],
    "legal_form": [legal_form_map[legal_form]],
    "region": [region_map[region]]
})

st.subheader("Model Input Preview")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Financial Health Class"):
    prediction_raw = model.predict(input_df)[0]

    class_map = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D"
    }

    prediction = class_map.get(prediction_raw, prediction_raw)

    st.subheader("Prediction Result")
    st.metric("Predicted Financial Health Class", prediction)

    if prediction == "D":
        st.error("High risk company. The company shows signs of financial distress.")
    elif prediction == "C":
        st.warning("Moderate risk company. The company should be monitored carefully.")
    elif prediction == "B":
        st.info("Stable company. Financial condition appears acceptable.")
    else:
        st.success("Healthy company. Financial indicators appear strong.")

    st.subheader("Business Interpretation")
    st.write(
        "The prediction is based on profitability, debt level, working capital, "
        "trend indicators, sector, legal form, and regional information. "
        "Higher debt, negative profit, weak working capital, and negative trend signals "
        "generally increase the probability of a weaker financial health class."
    )