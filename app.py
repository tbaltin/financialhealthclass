import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Financial Health Predictor", layout="wide")

st.title("Financial Health Classification and Early Warning System")

st.write(
    "Enter the company's financial data for the last three years. "
    "The system predicts the financial health class for each year and estimates the next year's risk direction."
)

model = joblib.load("xgboost_model.pkl")

class_order = {"A": 1, "B": 2, "C": 3, "D": 4}

def signed_log(x):
    return np.sign(x) * np.log1p(abs(x))

def to_float(value):
    try:
        return float(value)
    except:
        return 0.0

def map_class(pred):
    class_map = {0: "A", 1: "B", 2: "C", 3: "D", "A": "A", "B": "B", "C": "C", "D": "D"}
    return class_map.get(pred, pred)

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

st.subheader("Company Context")

col_a, col_b, col_c = st.columns(3)

with col_a:
    ateco_sector = st.selectbox("ATECO Sector", list(ateco_map.keys()))

with col_b:
    legal_form = st.selectbox("Legal Form", list(legal_form_map.keys()))

with col_c:
    region = st.selectbox("Region", list(region_map.keys()))

st.subheader("Financial Data for the Last Three Years")

years = ["Year 1", "Year 2", "Year 3"]
rows = []

for year in years:
    st.markdown(f"### {year}")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        net_profit = st.text_input(f"{year} - Net Profit / Loss (€)", "50000", key=f"profit_{year}")

    with c2:
        total_debt = st.text_input(f"{year} - Total Debt (€)", "400000", key=f"debt_{year}")

    with c3:
        current_assets = st.text_input(f"{year} - Current Assets (€)", "300000", key=f"assets_{year}")

    with c4:
        short_term_debt = st.text_input(f"{year} - Short-term Debt (€)", "150000", key=f"std_{year}")

    rows.append({
        "year": year,
        "net_profit": to_float(net_profit),
        "total_debt": to_float(total_debt),
        "current_assets": to_float(current_assets),
        "short_term_debt": to_float(short_term_debt)
    })

df = pd.DataFrame(rows)

df["working_capital"] = df["current_assets"] - df["short_term_debt"]
df["log_net_profit"] = df["net_profit"].apply(signed_log)
df["log_total_debt"] = df["total_debt"].apply(signed_log)

df["delta_profit"] = df["net_profit"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
df["delta_debt"] = df["total_debt"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

df["ateco_sector"] = ateco_map[ateco_sector]
df["legal_form"] = legal_form_map[legal_form]
df["region"] = region_map[region]

feature_cols = [
    "log_net_profit",
    "log_total_debt",
    "working_capital",
    "delta_profit",
    "delta_debt",
    "ateco_sector",
    "legal_form",
    "region"
]

input_df = df[feature_cols]

if st.button("Predict Classes and Next-Year Risk"):
    predictions_raw = model.predict(input_df)
    predictions = [map_class(p) for p in predictions_raw]

    df["Predicted Class"] = predictions

    st.subheader("Predicted Classes for the Last Three Years")

    display_df = df[[
        "year",
        "net_profit",
        "total_debt",
        "working_capital",
        "delta_profit",
        "delta_debt",
        "Predicted Class"
    ]]

    st.dataframe(display_df, use_container_width=True)

    last_class = predictions[-1]
    previous_classes = predictions[:-1]

    last_score = class_order[last_class]
    average_previous_score = np.mean([class_order[c] for c in previous_classes])

    profit_trend = df["net_profit"].iloc[-1] - df["net_profit"].iloc[0]
    debt_trend = df["total_debt"].iloc[-1] - df["total_debt"].iloc[0]
    wc_trend = df["working_capital"].iloc[-1] - df["working_capital"].iloc[0]

    next_risk_score = last_score

    if profit_trend < 0:
        next_risk_score += 0.5

    if debt_trend > 0:
        next_risk_score += 0.5

    if wc_trend < 0:
        next_risk_score += 0.5

    next_risk_score = min(4, round(next_risk_score))

    reverse_class_order = {1: "A", 2: "B", 3: "C", 4: "D"}
    next_year_class = reverse_class_order[next_risk_score]

    st.subheader("Estimated Next-Year Financial Health Class")

    st.metric("Next-Year Estimated Class", next_year_class)

    if class_order[next_year_class] > last_score:
        st.error("Warning: the next-year estimated class is weaker than the most recent year.")
    elif class_order[next_year_class] == last_score:
        st.info("The company is expected to maintain a similar financial health class.")
    else:
        st.success("The company is expected to improve its financial health class.")

    st.subheader("Business Recommendations")

    recommendations = []

    if profit_trend < 0:
        recommendations.append(
            "Profitability is weakening. The company should review cost structure, improve margins, and focus on more profitable business lines."
        )

    if debt_trend > 0:
        recommendations.append(
            "Debt is increasing. The company should reduce leverage, renegotiate debt terms, or avoid additional borrowing unless it directly improves profitability."
        )

    if wc_trend < 0:
        recommendations.append(
            "Working capital is deteriorating. The company should improve cash collection, reduce short-term liabilities, and strengthen liquidity buffers."
        )

    if class_order[next_year_class] > last_score:
        recommendations.append(
            "Because the predicted class is deteriorating, management should create an early-warning action plan with quarterly monitoring of debt, profit, and liquidity indicators."
        )

    if not recommendations:
        recommendations.append(
            "The company appears stable. To maintain or improve performance, it should preserve profitability, keep debt under control, and monitor liquidity regularly."
        )

    for rec in recommendations:
        st.write("- " + rec)

    st.subheader("Interpretation")

    st.write(
        "The system uses the last three years of profitability, debt level, working capital, "
        "profit trend, debt trend, sector, legal form, and region. If the company shows falling profit, "
        "rising debt, or weakening working capital, the next-year estimate becomes more cautious."
    )
