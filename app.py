import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Financial Health Predictor", layout="wide")

st.markdown("""
<style>
:root { color-scheme: light; }

.stApp {
    background: linear-gradient(135deg, #f7fbfc 0%, #eef7f8 45%, #ffffff 100%);
    color: #1f2933 !important;
}

html, body, [class*="css"], label, p, span, div {
    color: #1f2933 !important;
}

.main-title {
    background: linear-gradient(90deg, #23c7b7, #0c8fd3);
    padding: 32px;
    border-radius: 20px;
    color: white !important;
    margin-bottom: 30px;
}

.main-title h1, .main-title p {
    color: white !important;
}

.section-title {
    color: #009c9a !important;
    font-size: 25px;
    font-weight: 800;
    margin-top: 22px;
    margin-bottom: 14px;
}

.stButton > button {
    background: linear-gradient(90deg, #00b894, #0984c6);
    color: white !important;
    border-radius: 10px;
    font-weight: 700;
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
</style>
""", unsafe_allow_html=True)

model = joblib.load("xgboost_model.pkl")

def to_float(x):
    try:
        x = str(x).replace("−", "-").replace(",", "").replace("€", "").strip()
        return float(x)
    except:
        return 0.0

def signed_log(x):
    return np.sign(x) * np.log1p(abs(x))

ateco_map = {
    "01-03 Agriculture, Forestry, Fishing": 4,
    "05-09 Mining and Quarrying": 4,
    "10-33 Manufacturing": 2,
    "35 Electricity, Gas, Steam": 4,
    "36-39 Water Supply, Sewerage, Waste": 4,
    "41-43 Construction": 0,
    "45-47 Wholesale and Retail Trade": 3,
    "49-53 Transportation and Storage": 4,
    "55-56 Accommodation and Food Service": 4,
    "58-63 Information and Communication": 1,
    "64-66 Financial and Insurance Activities": 4,
    "68 Real Estate Activities": 4,
    "69-75 Professional, Scientific, Technical Activities": 4,
    "77-82 Administrative and Support Services": 4,
    "84 Public Administration": 4,
    "85 Education": 4,
    "86-88 Human Health and Social Work": 4,
    "90-93 Arts, Entertainment, Recreation": 4,
    "94-96 Other Service Activities": 4
}

legal_map = {
    "SRL": 0,
    "SPA": 1,
    "SNC": 2,
    "SAS": 2,
    "Individual Company": 2,
    "Cooperative": 2,
    "Consortium": 2,
    "Foundation": 2,
    "Association": 2,
    "Other": 2
}

region_map = {
    "Abruzzo": 2,
    "Aosta Valley": 2,
    "Apulia": 2,
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
    "Sardinia": 2,
    "Sicily": 2,
    "Trentino-Alto Adige": 2,
    "Tuscany": 2,
    "Umbria": 2,
    "Veneto": 2,
    "Other": 2
}

st.markdown("""
<div class="main-title">
    <h1>expert.ai Financial Health System</h1>
    <p>Predict company financial health and generate strategic recommendations.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Company Context</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    ateco = st.selectbox("ATECO Sector", list(ateco_map.keys()))

with c2:
    legal = st.selectbox("Legal Form", list(legal_map.keys()))

with c3:
    region = st.selectbox("Region", list(region_map.keys()))

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
        "Predicted Class": preds
    })

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True)

    final_class = preds[-1]

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
