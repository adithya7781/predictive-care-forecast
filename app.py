import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis import *

st.set_page_config(page_title="Care Load Forecasting Dashboard", layout="wide")

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("Department of Health & Human Services — Forecast Intelligence Dashboard")

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_pipeline():
    df = load_data("data/HHS_Unaccompanied_Alien_Children_Program.csv")
    df = create_forecasting_features(df)
    df = add_capacity_risk(df)
    return df

data = load_pipeline()

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header("Forecast Controls")

horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 30)

model_choice = st.sidebar.selectbox(
    "Select Forecast Model",
    ["Naive", "SARIMA", "Random Forest"]
)

# ======================================================
# SPLIT
# ======================================================

train, test = time_split(data, horizon)
actual = test["hhs_in_care"]

# ======================================================
# MODEL
# ======================================================

if model_choice == "Naive":
    preds = naive_forecast(train, horizon)
    conf = None

elif model_choice == "SARIMA":
    preds, conf = sarima_forecast(train, horizon)

else:
    preds = ml_forecast(train, test)
    conf = None

metrics = evaluate_forecast(actual, preds)

# ======================================================
# KPIs
# ======================================================

c1, c2, c3 = st.columns(3)
c1.metric("MAE", metrics["MAE"])
c2.metric("RMSE", metrics["RMSE"])
c3.metric("MAPE (%)", metrics["MAPE (%)"])

kpis = calculate_kpis(data)

c4, c5, c6, c7 = st.columns(4)
c4.metric("Avg Daily Intake", kpis["Avg Daily Intake"])
c5.metric("Avg Daily Discharge", kpis["Avg Daily Discharge"])
c6.metric("Pressure Days", kpis["Pressure Days"])
c7.metric("Capacity Breach Days", kpis["Capacity Breach Days"])

# ======================================================
# FORECAST CHART
# ======================================================

st.subheader("Forecast vs Actual")

fig = go.Figure()
fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name="Actual"))
fig.add_trace(go.Scatter(x=actual.index, y=preds, mode='lines', name="Forecast"))

if conf is not None:
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=conf.iloc[:, 0],
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=conf.iloc[:, 1],
        fill='tonexty',
        opacity=0.3,
        name="Confidence Interval"
    ))

st.plotly_chart(fig, width="stretch")

# ======================================================
# MODEL COMPARISON
# ======================================================

st.subheader("Model Comparison")

results = []

for m in ["Naive", "SARIMA", "Random Forest"]:

    if m == "Naive":
        p = naive_forecast(train, horizon)

    elif m == "SARIMA":
        p, _ = sarima_forecast(train, horizon)

    else:
        p = ml_forecast(train, test)

    r = evaluate_forecast(actual, p)
    results.append([m, r["MAE"], r["RMSE"], r["MAPE (%)"]])

comparison = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "MAPE"])
st.dataframe(comparison, width="stretch")
