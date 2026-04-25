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
# CUSTOM STYLE
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7fbfc 0%, #eef7f8 45%, #ffffff 100%);
    color: #1f2933;
}

.main-title {
    background: linear-gradient(90deg, #23c7b7, #0c8fd3);
    padding: 34px;
    border-radius: 20px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.10);
}

.main-title h1 {
    margin: 0;
    font-size: 44px;
    font-weight: 800;
}

.main-title p {
    font-size: 18px;
    margin-top: 8px;
}

.section-title {
    color: #009c9a;
    font-size: 26px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 16px;
}

.card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 26px;
}

.stButton > button {
    background: linear-gradient(90deg, #00b894, #0984c6);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 30px;
    font-weight: 800;
    font-size: 16px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0984c6, #00b894);
    color: white;
}

[data-testid="stMetric"] {
    background: #e9fbf7;
    border-radius: 18px;
    padding: 22px;
    border: 1px solid #b8eee4;
}

.success-box {
    background: #e9fbf1;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #00b894;
    color: #116149;
}

.warning-box {
    background: #fff6e6;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #f5a623;
    color: #6b4a00;
}

.danger-box {
    background: #ffecec;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #ff4d4f;
    color: #7a1f1f;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("xgboost_model.pkl")

# ===============================
# HELPERS
# ===============================
def to_float(value):
    try:
        value = str(value)
        value = value.replace("−", "-")
        value = value.replace(",", "")
        value = value.replace("€", "")
        value = value.strip()
        return float(value)
    except:
        return 0.0

def signed_log(x):
    return np.sign(x) * np.log1p(abs(x))

# ===============================
# MAPS
# ===============================
ateco_map = {
    "01-03: Agriculture, Forestry, Fishing": 4,
    "05-09: Mining and Quarrying": 4,
    "10-33: Manufacturing": 2,
    "35: Electricity, Gas, Steam": 4,
    "36-39: Water Supply, Sewerage, Waste": 4,
    "41-43: Construction": 0,
    "45-47: Wholesale and Retail Trade": 3,
    "49-53: Transportation and Storage": 4,
    "55-56: Accommodation and Food Service": 4,
    "58-63: Information and Communication": 1,
    "64-66: Financial and Insurance Activities": 4,
    "68: Real Estate Activities": 4,
    "69-75: Professional, Scientific, Technical Activities": 4,
    "77-82: Administrative and Support Services": 4,
    "84: Public Administration": 4,
    "85: Education": 4,
    "86-88: Human Health and Social Work": 4,
    "90-93: Arts, Entertainment, Recreation": 4,
    "94-96: Other Service Activities": 4
}

legal_form_map = {
    "SRL": 0,
    "SPA": 1,
    "SNC": 2,
    "SAS": 2,
    "Individual Company": 2,
    "Cooperative": 2,
    "Other": 2
}

region_map = {
    "Abruzzo": 2,
    "Basilicata": 2,
    "Calabria": 2,
    "Campania": 2,
    "Emilia-Romagna": 2,
    "Friuli-Venezia Giulia": 2,
    "Lazio": 0,
    "Liguria": 2,
    "Lombardy": 1,
    "Marche": 2,
    "Molise": 2,
    "Piedmont": 2,
    "Apulia": 2,
    "Sardinia": 2,
    "Sicily": 2,
    "Tuscany": 2,
    "Trentino-Alto Adige": 2,
    "Umbria": 2,
    "Aosta Valley": 2,
    "Veneto": 2,
    "Other": 2
}

# ===============================
# HEADER
# ===============================
col_logo, col_header = st.columns([1, 5])

with col_logo:
    st.image("expertai_logo.png", width=180)

with col_header:
    st.markdown("""
    <div class="main-title">
        <h1>Financial Health Prediction System</h1>
        <p>Predict company financial health and generate strategic recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# COMPANY CONTEXT
# ===============================
st.markdown('<div class="section-title">Company Context</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    ateco_sector = st.selectbox("ATECO Sector", list(ateco_map.keys()))

with col2:
    legal_form = st.selectbox("Legal Form", list(legal_form_map.keys()))

with col3:
    region = st.selectbox("Region", list(region_map.keys()))

# ===============================
# INPUT DATA
# ===============================
st.markdown('<div class="section-title">Enter Financial Data for the Last 3 Years</div>', unsafe_allow_html=True)

st.markdown("### Year 1")
col1, col2, col3, col4 = st.columns(4)
net_profit_1 = col1.text_input("Year 1 - Net Profit / Loss (€)", "")
total_debt_1 = col2.text_input("Year 1 - Total Debt (€)", "")
current_assets_1 = col3.text_input("Year 1 - Current Assets (€)", "")
short_debt_1 = col4.text_input("Year 1 - Short-term Debt (€)", "")

st.markdown("### Year 2")
col1, col2, col3, col4 = st.columns(4)
net_profit_2 = col1.text_input("Year 2 - Net Profit / Loss (€)", "")
total_debt_2 = col2.text_input("Year 2 - Total Debt (€)", "")
current_assets_2 = col3.text_input("Year 2 - Current Assets (€)", "")
short_debt_2 = col4.text_input("Year 2 - Short-term Debt (€)", "")

st.markdown("### Year 3")
col1, col2, col3, col4 = st.columns(4)
net_profit_3 = col1.text_input("Year 3 - Net Profit / Loss (€)", "")
total_debt_3 = col2.text_input("Year 3 - Total Debt (€)", "")
current_assets_3 = col3.text_input("Year 3 - Current Assets (€)", "")
short_debt_3 = col4.text_input("Year 3 - Short-term Debt (€)", "")

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("Predict Classes and Next-Year Risk"):

    net_profit_1 = to_float(net_profit_1)
    total_debt_1 = to_float(total_debt_1)
    current_assets_1 = to_float(current_assets_1)
    short_debt_1 = to_float(short_debt_1)

    net_profit_2 = to_float(net_profit_2)
    total_debt_2 = to_float(total_debt_2)
    current_assets_2 = to_float(current_assets_2)
    short_debt_2 = to_float(short_debt_2)

    net_profit_3 = to_float(net_profit_3)
    total_debt_3 = to_float(total_debt_3)
    current_assets_3 = to_float(current_assets_3)
    short_debt_3 = to_float(short_debt_3)

    def build_features(net_profit, total_debt, current_assets, short_debt, prev_profit=None, prev_debt=None):
        working_capital = current_assets - short_debt

        delta_profit = 0 if prev_profit is None else (net_profit - prev_profit) / (abs(prev_profit) + 1)
        delta_debt = 0 if prev_debt is None else (total_debt - prev_debt) / (abs(prev_debt) + 1)

        return [
            signed_log(net_profit),
            signed_log(total_debt),
            working_capital,
            delta_profit,
            delta_debt,
            ateco_map[ateco_sector],
            legal_form_map[legal_form],
            region_map[region]
        ]

    f1 = build_features(net_profit_1, total_debt_1, current_assets_1, short_debt_1)
    f2 = build_features(net_profit_2, total_debt_2, current_assets_2, short_debt_2, net_profit_1, total_debt_1)
    f3 = build_features(net_profit_3, total_debt_3, current_assets_3, short_debt_3, net_profit_2, total_debt_2)

    cols = [
        "log_net_profit",
        "log_total_debt",
        "working_capital",
        "delta_profit",
        "delta_debt",
        "ateco_sector",
        "legal_form",
        "region"
    ]

    df = pd.DataFrame([f1, f2, f3], columns=cols)

    preds_raw = model.predict(df)

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

    preds = [class_map.get(p, p) for p in preds_raw]

    result_df = pd.DataFrame({
        "Year": ["Year 1", "Year 2", "Year 3"],
        "Net Profit (€)": [net_profit_1, net_profit_2, net_profit_3],
        "Total Debt (€)": [total_debt_1, total_debt_2, total_debt_3],
        "Working Capital (€)": [
            current_assets_1 - short_debt_1,
            current_assets_2 - short_debt_2,
            current_assets_3 - short_debt_3
        ],
        "Predicted Class": preds
    })

    st.markdown('<div class="section-title">Predicted Classes</div>', unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True)

    st.markdown('<div class="section-title">Estimated Next-Year Class</div>', unsafe_allow_html=True)

    last_class = preds[-1]
    st.metric("Predicted Next Year", last_class)

    profit_change = net_profit_3 - net_profit_1
    debt_change = total_debt_3 - total_debt_1
    wc_1 = current_assets_1 - short_debt_1
    wc_3 = current_assets_3 - short_debt_3
    wc_change = wc_3 - wc_1

    st.markdown('<div class="section-title">Strategic Recommendations</div>', unsafe_allow_html=True)

    if last_class in ["A", "B"]:
        st.markdown("""
        <div class="success-box">
        <b>Company is financially stable.</b><br>
        The company should maintain discipline to preserve or improve its financial health.
        </div>
        """, unsafe_allow_html=True)

        st.write("""
        **How to maintain or improve performance:**
        - Keep debt growth under control and avoid unnecessary borrowing.
        - Maintain profitability by controlling costs and protecting margins.
        - Preserve strong working capital by improving cash collection and managing short-term liabilities.
        - Monitor key financial indicators every quarter.
        """)

        if debt_change > 0:
            st.info("Debt has increased over the three-year period. Rising debt should be monitored carefully.")

        if profit_change < 0:
            st.info("Profitability has weakened over the three-year period. The company should focus on margin improvement.")

        if wc_change < 0:
            st.info("Working capital has decreased. The company should strengthen liquidity to avoid future downgrades.")

    elif last_class == "C":
        st.markdown("""
        <div class="warning-box">
        <b>Moderate financial risk detected.</b><br>
        The company should act before the situation deteriorates further.
        </div>
        """, unsafe_allow_html=True)

        st.write("""
        **How to improve performance:**
        - Reduce leverage by limiting new debt.
        - Prioritise repayment of short-term obligations.
        - Improve profitability through cost reduction and more efficient operations.
        - Strengthen working capital by accelerating receivables collection.
        - Create a recovery plan with clear financial targets for the next year.
        """)

    elif last_class == "D":
        st.markdown("""
        <div class="danger-box">
        <b>High financial distress risk detected.</b><br>
        Immediate corrective action is recommended.
        </div>
        """, unsafe_allow_html=True)

        st.write("""
        **Urgent actions to improve performance:**
        - Restructure or renegotiate debt to reduce short-term pressure.
        - Restore profitability by cutting non-essential costs.
        - Focus on profitable business lines and reduce loss-making activities.
        - Improve liquidity by increasing cash reserves and reducing short-term liabilities.
        - Consider capital injection, asset sales, or operational restructuring.
        - Monitor the company monthly until risk indicators improve.
        """)
