import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
# ATECO MAP (FINAL)
# ===============================
ateco_map = {
    "10-33: Manufacturing": 2,
    "41-43: Construction": 0,
    "45-47: Wholesale and Retail Trade": 3,
    "58-63: Information and Communication": 1,
    "Other": 4
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
# UI
# ===============================
st.title("Financial Health Prediction System")

st.subheader("Company Context")

ateco_sector = st.selectbox("ATECO Sector", list(ateco_map.keys()))
legal_form = st.selectbox("Legal Form", list(legal_form_map.keys()))
region = st.selectbox("Region", list(region_map.keys()))

st.subheader("Enter Financial Data (Last 3 Years)")

# YEAR 1
st.markdown("### Year 1")
col1, col2, col3, col4 = st.columns(4)
net_profit_1 = col1.text_input("Net Profit (€)", "")
total_debt_1 = col2.text_input("Total Debt (€)", "")
current_assets_1 = col3.text_input("Current Assets (€)", "")
short_debt_1 = col4.text_input("Short-term Debt (€)", "")

# YEAR 2
st.markdown("### Year 2")
col1, col2, col3, col4 = st.columns(4)
net_profit_2 = col1.text_input("Net Profit (€) ", "")
total_debt_2 = col2.text_input("Total Debt (€) ", "")
current_assets_2 = col3.text_input("Current Assets (€) ", "")
short_debt_2 = col4.text_input("Short-term Debt (€) ", "")

# YEAR 3
st.markdown("### Year 3")
col1, col2, col3, col4 = st.columns(4)
net_profit_3 = col1.text_input("Net Profit (€)  ", "")
total_debt_3 = col2.text_input("Total Debt (€)  ", "")
current_assets_3 = col3.text_input("Current Assets (€)  ", "")
short_debt_3 = col4.text_input("Short-term Debt (€)  ", "")

# ===============================
# BUTTON
# ===============================
if st.button("Predict Classes and Next-Year Risk"):

    # convert inputs
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

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    def build_features(net_profit, total_debt, current_assets, short_debt,
                       prev_profit=None, prev_debt=None):

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

    # year features
    f1 = build_features(net_profit_1, total_debt_1, current_assets_1, short_debt_1)
    f2 = build_features(net_profit_2, total_debt_2, current_assets_2, short_debt_2,
                        net_profit_1, total_debt_1)
    f3 = build_features(net_profit_3, total_debt_3, current_assets_3, short_debt_3,
                        net_profit_2, total_debt_2)

    # feature names
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

    # ===============================
    # PREDICTION
    # ===============================
    preds = model.predict(df)

    class_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    preds = [class_map[p] for p in preds]

    # ===============================
    # OUTPUT
    # ===============================
    result_df = pd.DataFrame({
        "Year": ["Year 1", "Year 2", "Year 3"],
        "Net Profit": [net_profit_1, net_profit_2, net_profit_3],
        "Total Debt": [total_debt_1, total_debt_2, total_debt_3],
        "Working Capital": [
            current_assets_1 - short_debt_1,
            current_assets_2 - short_debt_2,
            current_assets_3 - short_debt_3
        ],
        "Predicted Class": preds
    })

    st.subheader("Predicted Classes")
    st.dataframe(result_df)

    # next year prediction = last year class
    st.subheader("Estimated Next-Year Class")
    st.success(f"Predicted Next Year: {preds[-1]}")

    # ===============================
    # SIMPLE RECOMMENDATION
    # ===============================
    st.subheader("Strategic Recommendations")

last_class = preds[-1]

profit_change = net_profit_3 - net_profit_1
debt_change = total_debt_3 - total_debt_1
wc_1 = current_assets_1 - short_debt_1
wc_3 = current_assets_3 - short_debt_3
wc_change = wc_3 - wc_1

if last_class in ["A", "B"]:
    st.success("✅ The company is financially stable, but it should maintain discipline to preserve or improve its class.")

    st.write("""
    **How to maintain or improve performance:**
    - Keep debt growth under control and avoid unnecessary borrowing.
    - Maintain positive profitability by protecting margins and controlling operating costs.
    - Preserve strong working capital by improving cash collection and managing short-term liabilities.
    - Monitor profitability, debt, and liquidity every quarter to detect early warning signals.
    """)

    if debt_change > 0:
        st.info("Debt has increased over the three-year period. Even though the company is currently stable, rising debt should be monitored carefully.")

    if profit_change < 0:
        st.info("Profitability has weakened over the three-year period. The company should focus on margin improvement and cost control.")

    if wc_change < 0:
        st.info("Working capital has decreased. The company should strengthen liquidity to avoid future downgrades.")

elif last_class == "C":
    st.warning("⚠️ The company shows moderate financial risk. It should act before the situation deteriorates further.")

    st.write("""
    **How to improve performance:**
    - Reduce leverage by limiting new debt and prioritising repayment of short-term obligations.
    - Improve profitability through cost reduction, pricing strategy, and more efficient operations.
    - Strengthen working capital by accelerating receivables collection and controlling inventories.
    - Build a recovery plan with clear financial targets for the next year.
    """)

elif last_class == "D":
    st.error("🚨 The company is predicted as financially distressed. Immediate corrective action is recommended.")

    st.write("""
    **Urgent actions to improve performance:**
    - Restructure or renegotiate debt to reduce short-term pressure.
    - Restore profitability by cutting non-essential costs and focusing on profitable activities.
    - Improve liquidity by increasing cash reserves and reducing short-term liabilities.
    - Consider capital injection, asset sales, or operational restructuring if financial pressure continues.
    - Monitor the company monthly rather than yearly until risk indicators improve.
    """)
