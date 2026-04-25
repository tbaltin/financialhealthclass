import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Financial Health Predictor",
    layout="wide"
)

# ===============================
# CUSTOM STYLE (FIXED)
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7fbfc 0%, #eef7f8 45%, #ffffff 100%);
    color: #1f2933;
}

.main-title {
    background: linear-gradient(90deg, #23c7b7, #0c8fd3);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 30px;
}

.section-title {
    color: #009c9a;
    font-size: 24px;
    font-weight: 800;
    margin-top: 20px;
}

.success-box {
    background: #e9fbf1;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #00b894;
}

.warning-box {
    background: #fff6e6;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #f5a623;
}

.danger-box {
    background: #ffecec;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #ff4d4f;
}

.stButton > button {
    background: linear-gradient(90deg, #00b894, #0984c6);
    color: white;
    border-radius: 10px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/*  FORCE TEXT COLOR FIX */
html, body, [class*="css"]  {
    color: #1f2933 !important;
}

/* Input label fix */
label {
    color: #1f2933 !important;
    font-weight: 600;
}

/* Selectbox text */
div[data-baseweb="select"] * {
    color: #1f2933 !important;
}

/* Input text */
input {
    color: #1f2933 !important;
}

/* Placeholder */
input::placeholder {
    color: #6b7280 !important;
}

/* Dropdown menu */
ul {
    color: #1f2933 !important;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER (NO IMAGE ERROR)
# ===============================
st.markdown("""
<div class="main-title">
    <h1>expert.ai Financial Health System</h1>
    <p>Predict company financial health and generate smart recommendations</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("xgboost_model.pkl")

# ===============================
# HELPERS
# ===============================
def to_float(x):
    try:
        return float(str(x).replace(",", "").replace("€", ""))
    except:
        return 0.0

def signed_log(x):
    return np.sign(x) * np.log1p(abs(x))

# ===============================
# CATEGORY MAPS
# ===============================
ateco_map = {
    "10-33 Manufacturing": 2,
    "41-43 Construction": 0,
    "45-47 Wholesale Retail": 3,
    "58-63 IT & Communication": 1,
    "Other": 4
}

legal_map = {
    "SRL": 0,
    "SPA": 1,
    "Other": 2
}

region_map = {
    "Lazio": 0,
    "Lombardy": 1,
    "Other": 2
}

# ===============================
# INPUT UI
# ===============================
st.markdown('<div class="section-title">Company Context</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    ateco = st.selectbox("ATECO Sector", list(ateco_map.keys()))

with col2:
    legal = st.selectbox("Legal Form", list(legal_map.keys()))

with col3:
    region = st.selectbox("Region", list(region_map.keys()))

# ===============================
# FINANCIAL INPUTS
# ===============================
st.markdown('<div class="section-title">Financial Data (Last 3 Years)</div>', unsafe_allow_html=True)

def year_inputs(label):
    c1, c2, c3, c4 = st.columns(4)
    return (
        to_float(c1.text_input(f"{label} Net Profit (€)")),
        to_float(c2.text_input(f"{label} Total Debt (€)")),
        to_float(c3.text_input(f"{label} Current Assets (€)")),
        to_float(c4.text_input(f"{label} Short-term Debt (€)")),
    )

y1 = year_inputs("Year 1")
y2 = year_inputs("Year 2")
y3 = year_inputs("Year 3")

# ===============================
# FEATURE BUILDER
# ===============================
def build_features(curr, prev=None):
    net, debt, assets, short = curr
    wc = assets - short

    if prev:
        prev_net, prev_debt, _, _ = prev
        d_profit = (net - prev_net) / (abs(prev_net) + 1)
        d_debt = (debt - prev_debt) / (abs(prev_debt) + 1)
    else:
        d_profit, d_debt = 0, 0

    return [
        signed_log(net),
        signed_log(debt),
        wc,
        d_profit,
        d_debt,
        ateco_map[ateco],
        legal_map[legal],
        region_map[region]
    ]

# ===============================
# PREDICT
# ===============================
if st.button("Predict"):

    f1 = build_features(y1)
    f2 = build_features(y2, y1)
    f3 = build_features(y3, y2)

    df = pd.DataFrame([f1, f2, f3], columns=[
        "log_net_profit",
        "log_total_debt",
        "working_capital",
        "delta_profit",
        "delta_debt",
        "ateco_sector",
        "legal_form",
        "region"
    ])

    preds = model.predict(df)

    class_map = {0:"A",1:"B",2:"C",3:"D"}
    preds = [class_map[p] for p in preds]

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        "Year": ["Y1","Y2","Y3"],
        "Class": preds
    }))

    final = preds[-1]

    st.markdown('<div class="section-title">Next Year Prediction</div>', unsafe_allow_html=True)
    st.metric("Predicted Class", final)

    # ===============================
    # SMART RECOMMENDATIONS
    # ===============================
    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

    if final in ["A","B"]:
        st.markdown("""
        <div class="success-box">
        Company is stable. Maintain performance by:
        <ul>
        <li>Controlling debt growth</li>
        <li>Keeping profitability stable</li>
        <li>Maintaining strong liquidity</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif final == "C":
        st.markdown("""
        <div class="warning-box">
        Moderate risk detected. Improve by:
        <ul>
        <li>Reducing debt levels</li>
        <li>Improving profit margins</li>
        <li>Strengthening working capital</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="danger-box">
        High financial distress risk.
        <ul>
        <li>Restructure debt immediately</li>
        <li>Cut unnecessary costs</li>
        <li>Increase liquidity</li>
        <li>Focus on profitable activities</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
