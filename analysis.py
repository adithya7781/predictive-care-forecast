import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ======================================================
# DATA LOADING & VALIDATION
# ======================================================

def load_data(file_path):
    """
    Load UAC dataset and perform basic validation
    """

    df = pd.read_csv(file_path)

    # Rename columns cleanly
    df.columns = [
        "date",
        "cbp_apprehended",
        "cbp_in_care",
        "transferred_to_hhs",
        "hhs_in_care",
        "discharged"
    ]

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert types
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    # Ensure daily continuity
    df = df.asfreq("D")
    df.interpolate(method="linear", inplace=True)

    return df


# ======================================================
# TIME SERIES DECOMPOSITION FEATURES
# ======================================================

def add_time_features(df):
    """
    Adds calendar-based predictive signals
    """

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["week"] = df.index.isocalendar().week

    return df


# ======================================================
# FEATURE ENGINEERING FOR FORECASTING
# ======================================================

def create_forecasting_features(df):
    """
    Creates lag, rolling, and pressure indicators
    """

    # Lag features
    df["lag_1"] = df["hhs_in_care"].shift(1)
    df["lag_7"] = df["hhs_in_care"].shift(7)
    df["lag_14"] = df["hhs_in_care"].shift(14)

    # Rolling stats
    df["roll_mean_7"] = df["hhs_in_care"].rolling(7).mean()
    df["roll_mean_14"] = df["hhs_in_care"].rolling(14).mean()
    df["roll_std_7"] = df["hhs_in_care"].rolling(7).std()

    # Flow pressure
    df["net_pressure"] = df["transferred_to_hhs"] - df["discharged"]

    df.dropna(inplace=True)
    return df


# ======================================================
# TRAIN TEST SPLIT (TIME BASED)
# ======================================================

def time_split(df, horizon=30):
    """
    Strict time-based split
    """
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    return train, test


# ======================================================
# BASELINE MODELS
# ======================================================

def naive_forecast(train, horizon):
    """
    Last-value persistence model
    """
    last_val = train["hhs_in_care"].iloc[-1]
    return np.repeat(last_val, horizon)


def moving_average_forecast(train, horizon, window=7):
    """
    Moving average baseline
    """
    avg = train["hhs_in_care"].rolling(window).mean().iloc[-1]
    return np.repeat(avg, horizon)


# ======================================================
# ARIMA / SARIMA FORECAST
# ======================================================

def sarima_forecast(train, horizon):
    """
    Statistical forecasting with confidence intervals
    """

    model = SARIMAX(
        train["hhs_in_care"],
        order=(1,1,1),
        seasonal_order=(1,1,1,7)
    )

    result = model.fit(disp=False)

    forecast_obj = result.get_forecast(steps=horizon)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    return forecast, conf_int


# ======================================================
# MACHINE LEARNING FORECAST MODELS
# ======================================================

def ml_forecast(train, test, model_type="rf"):
    """
    Random Forest / Gradient Boosting forecasting
    """

    X_train = train.drop("hhs_in_care", axis=1)
    y_train = train["hhs_in_care"]

    X_test = test.drop("hhs_in_care", axis=1)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        model = GradientBoostingRegressor(random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return preds


# ======================================================
# MODEL EVALUATION METRICS
# ======================================================

def evaluate_forecast(actual, predicted):
    """
    MAE, RMSE, MAPE evaluation
    """

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2)
    }


# ======================================================
# CAPACITY RISK & EARLY WARNING KPIs
# ======================================================

def add_capacity_risk(df):
    """
    Early warning indicator for overcrowding risk
    """

    df["capacity_status"] = np.where(
        df["hhs_in_care"] > 15000,
        "Critical",
        np.where(df["hhs_in_care"] > 12000, "Warning", "Normal")
    )

    return df


# ======================================================
# FORECAST KPI CALCULATIONS
# ======================================================

def calculate_kpis(df):
    """
    Computes decision-making KPIs
    """

    avg_intake = df["transferred_to_hhs"].mean()
    avg_discharge = df["discharged"].mean()

    pressure_days = (df["net_pressure"] > 0).sum()
    capacity_breach_days = (df["capacity_status"] == "Critical").sum()

    kpis = {
        "Avg Daily Intake": round(avg_intake, 2),
        "Avg Daily Discharge": round(avg_discharge, 2),
        "Pressure Days": int(pressure_days),
        "Capacity Breach Days": int(capacity_breach_days)
    }

    return kpis
