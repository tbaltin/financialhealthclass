import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Financial Health Predictor", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7fbfc 0%, #eef7f8 50%, #ffffff 100%);
}

/* Main header */
.main-title {
    background: linear-gradient(90deg, #23c7b7, #0c8fd3);
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 30px;
}

.main-title h1,
.main-title p {
    color: white !important;
}

/* Section titles */
.section-title {
    color: #009c9a !important;
    font-size: 25px;
    font-weight: 800;
    margin-top: 22px;
    margin-bottom: 14px;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #00b894, #0984c6);
    color: white !important;
    border-radius: 10px;
    font-weight: 700;
}

/* Recommendation boxes with readable text */
.success-box {
    background: #e9fbf1;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #00b894;
    color: #12372a !important;
}

.warning-box {
    background: #fff6e6;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #f5a623;
    color: #4a3200 !important;
}

.danger-box {
    background: #ffecec;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #ff4d4f;
    color: #5c1717 !important;
}

.success-box * {
    color: #12372a !important;
}

.warning-box * {
    color: #4a3200 !important;
}

.danger-box * {
    color: #5c1717 !important;
}

/* Hide Streamlit toolbar */
[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0%;
}

[data-testid="stDecoration"] {
    display: none;
}

[data-testid="stStatusWidget"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD OLD MODEL
# ===============================
model = joblib.load("xgboost_model.pkl")

# ===============================
# HELPERS
# ===============================
def to_float(x):
    try:
        x = str(x).replace("−", "-").replace(",", "").replace("€", "").strip()
        return float(x)
    except:
        return 0.0

def signed_log(x):
    return np.sign(x) * np.log1p(abs(x))

# ===============================
# MAPS
# ===============================
ateco_map = {
    "43 — Specialised Construction": 0,
    "46 — Wholesale Trade": 3,
    "47 — Retail Trade": 3,
    "IT / Information Services": 1,
    "Other": 4
}

legal_map = {
    "SRL": 0,
    "SPA": 1,
    "SNC": 2,
    "SAS": 2,
    "Individual Company": 2,
    "Cooperative": 2,
    "Other": 2
}

region_map = {
    "Lazio": 0,
    "Lombardy": 1,
    "Abruzzo": 2,
    "Aosta Valley": 2,
    "Apulia": 2,
    "Basilicata": 2,
    "Calabria": 2,
    "Campania": 2,
    "Emilia-Romagna": 2,
    "Friuli-Venezia Giulia": 2,
    "Liguria": 2,
    "Marche": 2,
    "Molise": 2,
    "Piedmont": 2,
    "Sardinia": 2,
    "Sicily": 2,
    "Trentino-Alto Adige": 2,
    "Tuscany": 2,
    "Umbria": 2,
    "Veneto": 2,
    "Other": 2
}

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="main-title">
    <h1>expert.ai Financial Health System</h1>
    <p>Predict company financial health and generate strategic recommendations.</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# COMPANY CONTEXT
# ===============================
st.markdown('<div class="section-title">Company Context</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    ateco = st.selectbox("ATECO Sector", list(ateco_map.keys()))

with c2:
    legal = st.selectbox("Legal Form", list(legal_map.keys()))

with c3:
    region = st.selectbox("Region", list(region_map.keys()))

# ===============================
# INPUTS
# ===============================
st.markdown('<div class="section-title">Financial Data for the Last 3 Years</div>', unsafe_allow_html=True)

def year_inputs(year):
    st.markdown(f"### {year}")
    a, b, c, d = st.columns(4)

    net_profit = a.text_input(f"{year} - Net Profit / Loss (€)", "")
    total_debt = b.text_input(f"{year} - Total Debt (€)", "")
    current_assets = c.text_input(f"{year} - Current Assets (€)", "")
    short_debt = d.text_input(f"{year} - Short-term Debt (€)", "")

    return (
        to_float(net_profit),
        to_float(total_debt),
        to_float(current_assets),
        to_float(short_debt)
    )

y1 = year_inputs("Year 1")
y2 = year_inputs("Year 2")
y3 = year_inputs("Year 3")

# ===============================
# FEATURE ENGINEERING FOR OLD MODEL
# ===============================
def build_features(current, previous=None):
    net_profit, total_debt, current_assets, short_debt = current

    working_capital = current_assets - short_debt

    if previous is None:
        delta_profit = 0
        delta_debt = 0
    else:
        prev_profit, prev_debt, _, _ = previous

        delta_profit = (net_profit - prev_profit) / (abs(prev_profit) + 1)
        delta_debt = (total_debt - prev_debt) / (abs(prev_debt) + 1)

    return [
        signed_log(net_profit),
        signed_log(total_debt),
        working_capital,
        delta_profit,
        delta_debt,
        ateco_map[ateco],
        legal_map[legal],
        region_map[region]
    ]

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Classes and Next-Year Risk"):

    feature_columns = [
        "log_net_profit",
        "log_total_debt",
        "working_capital",
        "delta_profit",
        "delta_debt",
        "ateco_sector",
        "legal_form",
        "region"
    ]

    input_df = pd.DataFrame([
        build_features(y1),
        build_features(y2, y1),
        build_features(y3, y2)
    ], columns=feature_columns)

    raw_preds = model.predict(input_df)

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

    preds = [class_map.get(p, p) for p in raw_preds]

    result_df = pd.DataFrame({
        "Year": ["Year 1", "Year 2", "Year 3"],
        "Net Profit (€)": [y1[0], y2[0], y3[0]],
        "Total Debt (€)": [y1[1], y2[1], y3[1]],
        "Working Capital (€)": [
            y1[2] - y1[3],
            y2[2] - y2[3],
            y3[2] - y3[3]
        ],
        "Model-Predicted Class": preds
    })

    st.markdown('<div class="section-title">Model-Predicted Classes</div>', unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True)

    # ===============================
    # NEXT-YEAR PROJECION
    # ===============================

    next_net_profit = y3[0] + (y3[0] - y2[0])
    next_total_debt = y3[1] + (y3[1] - y2[1])
    next_current_assets = y3[2] + (y3[2] - y2[2])
    next_short_debt = y3[3] + (y3[3] - y2[3])

    next_year = (
        next_net_profit,
        next_total_debt,
        next_current_assets,
        next_short_debt
    )

    next_features = pd.DataFrame(
        [build_features(next_year, y3)],
        columns=feature_columns
    )

    next_raw_pred = model.predict(next_features)[0]
    final_class = class_map.get(next_raw_pred, next_raw_pred)

    st.markdown('<div class="section-title">Estimated Next-Year Risk Class</div>', unsafe_allow_html=True)
    st.metric("Estimated Class Based on Recent Trend", final_class)

    profit_change = y3[0] - y1[0]
    debt_change = y3[1] - y1[1]
    wc_change = (y3[2] - y3[3]) - (y1[2] - y1[3])

    st.markdown('<div class="section-title">Strategic Recommendations</div>', unsafe_allow_html=True)

    if final_class in ["A", "B"]:
        st.markdown("""
        <div class="success-box">
            <b>The company appears financially stable.</b><br>
            The priority is to maintain performance and avoid future deterioration.
        </div>
        """, unsafe_allow_html=True)

        st.write("""
        **Recommended actions:**
        - Keep debt growth under control and avoid unnecessary borrowing.
        - Maintain profitability by controlling costs and protecting margins.
        - Preserve strong working capital through better cash collection and short-term liability management.
        - Monitor profit, debt, and liquidity indicators every quarter.
        """)

        if debt_change > 0:
            st.info("Debt increased over the three-year period. This should be monitored even if the company is currently stable.")

        if profit_change < 0:
            st.info("Profitability decreased over the three-year period. The company should focus on margin improvement.")

        if wc_change < 0:
            st.info("Working capital decreased. The company should strengthen liquidity buffers.")

    elif final_class == "C":
        st.markdown("""
        <div class="warning-box">
            <b>Moderate financial risk detected.</b><br>
            The company should act before the situation deteriorates.
        </div>
        """, unsafe_allow_html=True)

        st.write("""
        **Recommended actions:**
        - Reduce leverage by limiting new debt.
        - Improve profitability through cost control and operational efficiency.
        - Strengthen working capital by accelerating receivables collection.
        - Create a one-year recovery plan with measurable financial targets.
        """)

    else:
        st.markdown("""
        <div class="danger-box">
            <b>High financial distress risk detected.</b><br>
            Immediate corrective action is recommended.
        </div>
        """, unsafe_allow_html=True)

        st.write("""
        **Recommended actions:**
        - Restructure or renegotiate debt to reduce short-term pressure.
        - Cut non-essential costs and focus on profitable activities.
        - Improve liquidity by increasing cash reserves and reducing short-term liabilities.
        - Consider capital injection, asset sales, or operational restructuring.
        - Monitor financial indicators monthly until risk improves.
        """)
