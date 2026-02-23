import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis import *

# ======================================================
# UI THEME STYLE
# ======================================================

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.3);
}

div[data-testid="metric-container"] > label {
    font-size: 14px;
    color: #C9D1D9;
}

div[data-testid="metric-container"] > div {
    font-size: 26px;
    font-weight: bold;
    color: #58A6FF;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Care Load Forecasting Dashboard",
    layout="wide"
)

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("Department of Health & Human Services — Forecast Intelligence Dashboard")

# ======================================================
# DATA PIPELINE (CACHED)
# ======================================================

@st.cache_data
def load_full_pipeline():
    df = load_data("data/HHS_Unaccompanied_Alien_Children_Program.csv")
    df = add_time_features(df)
    df = create_forecasting_features(df)
    df = add_capacity_risk(df)
    return df

with st.spinner("Loading forecasting intelligence system..."):
    data = load_full_pipeline()

# ======================================================
# SIDEBAR CONTROLS
# ======================================================

st.sidebar.header("Forecast Controls")

horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 30)

model_choice = st.sidebar.selectbox(
    "Select Forecast Model",
    ["Naive", "Moving Average", "SARIMA", "Random Forest", "Gradient Boosting"]
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [data.index.min(), data.index.max()]
)

# ======================================================
# FILTER DATA
# ======================================================

filtered = data[
    (data.index >= pd.to_datetime(date_range[0])) &
    (data.index <= pd.to_datetime(date_range[1]))
]

train, test = time_split(filtered, horizon)
actual = test["hhs_in_care"]

# ======================================================
# MODEL EXECUTION
# ======================================================

if model_choice == "Naive":
    preds = naive_forecast(train, horizon)
    conf = None

elif model_choice == "Moving Average":
    preds = moving_average_forecast(train, horizon)
    conf = None

elif model_choice == "SARIMA":
    preds, conf = sarima_forecast(train, horizon)

elif model_choice == "Random Forest":
    preds = ml_forecast(train, test, "rf")
    conf = None

else:
    preds = ml_forecast(train, test, "gb")
    conf = None

metrics_result = evaluate_forecast(actual, preds)

# ======================================================
# KPI DISPLAY
# ======================================================

k1, k2, k3 = st.columns(3)
k1.metric("MAE", metrics_result["MAE"])
k2.metric("RMSE", metrics_result["RMSE"])
k3.metric("MAPE (%)", metrics_result["MAPE (%)"])

decision_kpis = calculate_kpis(filtered)

k4, k5, k6, k7 = st.columns(4)
k4.metric("Avg Daily Intake", decision_kpis["Avg Daily Intake"])
k5.metric("Avg Daily Discharge", decision_kpis["Avg Daily Discharge"])
k6.metric("Pressure Days", decision_kpis["Pressure Days"])
k7.metric("Capacity Breach Days", decision_kpis["Capacity Breach Days"])

# ======================================================
# EXECUTIVE SUMMARY
# ======================================================

st.markdown("## Executive Intelligence Summary")

st.info("""
### Healthcare Capacity Intelligence

• Forecasting models estimate future HHS care load and discharge demand.

• Net pressure indicators reveal imbalance between intake and placements.

• Capacity breach warnings allow proactive staffing and shelter planning.

• Early-warning signals help prevent overcrowding and burnout.

### Strategic Recommendations

✔ Scale shelters and staff before projected surge  
✔ Increase discharge placement processing  
✔ Monitor net pressure daily  
✔ Use predictive dashboard for planning decisions  
""")

# ======================================================
# FORECAST VISUALIZATION
# ======================================================

st.subheader("Future Care Load Forecast")

fig = go.Figure()
fig.add_trace(go.Scatter(x=actual.index, y=actual.values,
                         mode='lines', name='Actual'))

fig.add_trace(go.Scatter(x=actual.index, y=preds,
                         mode='lines', name='Forecast'))

if conf is not None:
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=conf.iloc[:,0],
        fill=None,
        mode='lines',
        line_color='lightgrey',
        showlegend=False))

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=conf.iloc[:,1],
        fill='tonexty',
        mode='lines',
        line_color='lightgrey',
        name='Confidence Interval'))

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# INTAKE VS DISCHARGE
# ======================================================

st.subheader("Intake vs Discharge Trend")

flow_fig = px.line(
    filtered,
    y=["transferred_to_hhs","discharged"],
    title="Daily Intake vs Discharge"
)

st.plotly_chart(flow_fig, use_container_width=True)

# ======================================================
# NET PRESSURE ANALYSIS
# ======================================================

st.subheader("Net Pressure Indicator")

pressure_chart = px.line(
    filtered,
    y="net_pressure",
    title="System Pressure (Transfers − Discharges)"
)

st.plotly_chart(pressure_chart, use_container_width=True)

# ======================================================
# CAPACITY RISK PANEL
# ======================================================

st.subheader("Capacity Risk Monitoring")

risk_chart = px.pie(
    filtered,
    names="capacity_status",
    title="Capacity Risk Distribution"
)

st.plotly_chart(risk_chart, use_container_width=True)

risk_table = filtered[["hhs_in_care","capacity_status"]].tail(15)
st.dataframe(risk_table, use_container_width=True)

# ======================================================
# MODEL COMPARISON (ADVANCED)
# ======================================================

st.subheader("Model Comparison")

model_results = []

for m in ["Naive","Moving Average","SARIMA","Random Forest","Gradient Boosting"]:

    if m == "Naive":
        p = naive_forecast(train, horizon)
    elif m == "Moving Average":
        p = moving_average_forecast(train, horizon)
    elif m == "SARIMA":
        p,_ = sarima_forecast(train, horizon)
    elif m == "Random Forest":
        p = ml_forecast(train, test, "rf")
    else:
        p = ml_forecast(train, test, "gb")

    r = evaluate_forecast(actual, p)
    model_results.append([m, r["MAE"], r["RMSE"], r["MAPE (%)"]])

comparison_df = pd.DataFrame(
    model_results,
    columns=["Model","MAE","RMSE","MAPE"]
)

st.dataframe(comparison_df, use_container_width=True)
