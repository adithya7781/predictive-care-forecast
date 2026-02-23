import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis import *

st.set_page_config(page_title="Care Forecast Dashboard", layout="wide")

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("HHS Forecast Intelligence Dashboard")

# =========================
# LOAD
# =========================
@st.cache_data
def load_pipeline():
    df = load_data("data/HHS_Unaccompanied_Alien_Children_Program.csv")
    df = create_forecasting_features(df)
    df = add_capacity_risk(df)
    return df

data = load_pipeline()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Forecast Controls")

horizon = st.sidebar.slider("Forecast Horizon", 7, 60, 30)

model_choice = st.sidebar.selectbox(
    "Model",
    ["SARIMA", "Random Forest", "Naive"]
)

date_range = st.sidebar.date_input(
    "Date Range",
    [data.index.min(), data.index.max()]
)

# =========================
# FILTER
# =========================
filtered = data[
    (data.index >= pd.to_datetime(date_range[0])) &
    (data.index <= pd.to_datetime(date_range[1]))
]

min_required = horizon + 30
if len(filtered) < min_required:
    st.warning("Selected date range too small for forecasting. Select larger range.")
    st.stop()

train, test = time_split(filtered, horizon)
actual = test["hhs_in_care"]

# =========================
# MODEL RUN
# =========================
if model_choice == "SARIMA":
    preds, conf = sarima_forecast(train, horizon)

elif model_choice == "Random Forest":
    preds = ml_forecast(train, test)
    conf = None

    # Feature importance check (ONLY for RF)
    from sklearn.ensemble import RandomForestRegressor
    X_train = train.drop("hhs_in_care", axis=1).select_dtypes(include="number")
    y_train = train["hhs_in_care"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    importance = pd.Series(model.feature_importances_, index=X_train.columns)\
        .sort_values(ascending=False)

    st.subheader("Feature Importance")
    st.dataframe(importance)

else:
    preds = naive_forecast(train, horizon)
    conf = None

metrics = evaluate_forecast(actual, preds)

# =========================
# KPI ROW
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("MAE", metrics["MAE"])
c2.metric("RMSE", metrics["RMSE"])
c3.metric("MAPE", metrics["MAPE (%)"])

kpi = calculate_kpis(filtered)

c4, c5, c6, c7 = st.columns(4)
c4.metric("Avg Intake", kpi["Avg Daily Intake"])
c5.metric("Avg Discharge", kpi["Avg Daily Discharge"])
c6.metric("Pressure Days", kpi["Pressure Days"])
c7.metric("Capacity Breach", kpi["Capacity Breach Days"])

# =========================
# FORECAST CHART
# =========================
st.subheader("Future Care Load Forecast")

fig = go.Figure()
fig.add_trace(go.Scatter(x=actual.index, y=actual, name="Actual"))
fig.add_trace(go.Scatter(x=actual.index, y=preds, name="Forecast"))

if conf is not None:
    fig.add_trace(go.Scatter(x=actual.index, y=conf.iloc[:, 0], line=dict(width=0)))
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=conf.iloc[:, 1],
        fill='tonexty',
        opacity=0.3,
        name="Confidence"
    ))

st.plotly_chart(fig, width="stretch")

# =========================
# INTAKE VS DISCHARGE
# =========================
st.subheader("Intake vs Discharge")
flow = px.line(filtered, y=["transferred_to_hhs", "discharged"])
st.plotly_chart(flow, width="stretch")

# =========================
# PRESSURE
# =========================
st.subheader("Net Pressure")
press = px.line(filtered, y="net_pressure")
st.plotly_chart(press, width="stretch")

# =========================
# CAPACITY
# =========================
st.subheader("Capacity Risk")
pie = px.pie(filtered, names="capacity_status")
st.plotly_chart(pie, width="stretch")

st.dataframe(filtered.tail(20), width="stretch")

# =========================
# MODEL COMPARISON
# =========================
st.subheader("Model Comparison")

rows = []

for m in ["SARIMA", "Random Forest", "Naive"]:
    if m == "SARIMA":
        p, _ = sarima_forecast(train, horizon)
    elif m == "Random Forest":
        p = ml_forecast(train, test)
    else:
        p = naive_forecast(train, horizon)

    r = evaluate_forecast(actual, p)
    rows.append([m, r["MAE"], r["RMSE"], r["MAPE (%)"]])

st.dataframe(pd.DataFrame(rows,
                          columns=["Model", "MAE", "RMSE", "MAPE"]),
             width="stretch")
