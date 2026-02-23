import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis import *

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
# LOAD PIPELINE
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
# SIDEBAR
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

# avoid crash if small data
if len(filtered) < horizon + 5:
    st.error("Not enough data for selected horizon. Reduce horizon.")
    st.stop()

train, test = time_split(filtered, horizon)
actual = test["hhs_in_care"]

# ======================================================
# MODEL RUN
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

c1, c2, c3 = st.columns(3)
c1.metric("MAE", metrics_result["MAE"])
c2.metric("RMSE", metrics_result["RMSE"])
c3.metric("MAPE (%)", metrics_result["MAPE (%)"])

decision_kpis = calculate_kpis(filtered)

c4, c5, c6, c7 = st.columns(4)
c4.metric("Avg Daily Intake", decision_kpis["Avg Daily Intake"])
c5.metric("Avg Daily Discharge", decision_kpis["Avg Daily Discharge"])
c6.metric("Pressure Days", decision_kpis["Pressure Days"])
c7.metric("Capacity Breach Days", decision_kpis["Capacity Breach Days"])

# ======================================================
# FORECAST CHART
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
        y=conf.iloc[:, 0],
        line=dict(width=0),
        showlegend=False))

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=conf.iloc[:, 1],
        fill='tonexty',
        name="Confidence Interval",
        opacity=0.3))

st.plotly_chart(fig, width="stretch")

# ======================================================
# INTAKE VS DISCHARGE
# ======================================================

st.subheader("Intake vs Discharge Trend")

flow_fig = px.line(
    filtered,
    y=["transferred_to_hhs", "discharged"],
    title="Daily Intake vs Discharge"
)

st.plotly_chart(flow_fig, width="stretch")

# ======================================================
# NET PRESSURE
# ======================================================

st.subheader("Net Pressure Indicator")

pressure_chart = px.line(
    filtered,
    y="net_pressure",
    title="System Pressure (Transfers − Discharges)"
)

st.plotly_chart(pressure_chart, width="stretch")

# ======================================================
# CAPACITY RISK
# ======================================================

st.subheader("Capacity Risk Monitoring")

risk_chart = px.pie(
    filtered,
    names="capacity_status",
    title="Capacity Risk Distribution"
)

st.plotly_chart(risk_chart, width="stretch")

st.dataframe(
    filtered[["hhs_in_care", "capacity_status"]].tail(15),
    width="stretch"
)

# ======================================================
# MODEL COMPARISON (FIXED)
# ======================================================

st.subheader("Model Comparison")

model_results = []

models = ["Naive", "Moving Average", "SARIMA", "Random Forest", "Gradient Boosting"]

for m in models:
    try:
        if m == "Naive":
            p = naive_forecast(train, horizon)

        elif m == "Moving Average":
            p = moving_average_forecast(train, horizon)

        elif m == "SARIMA":
            p, _ = sarima_forecast(train, horizon)

        elif m == "Random Forest":
            p = ml_forecast(train, test, "rf")

        else:
            p = ml_forecast(train, test, "gb")

        r = evaluate_forecast(actual, p)
        model_results.append([m, r["MAE"], r["RMSE"], r["MAPE (%)"]])

    except Exception as e:
        model_results.append([m, "Error", "Error", "Error"])

comparison_df = pd.DataFrame(
    model_results,
    columns=["Model", "MAE", "RMSE", "MAPE"]
)
st.dataframe(comparison_df, width="stretch"
