import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ======================================================
# DATA LOADING & CLEANING
# ======================================================

def load_data(file_path):

    df = pd.read_csv(file_path)

    df.columns = [
        "date",
        "cbp_apprehended",
        "cbp_in_care",
        "transferred_to_hhs",
        "hhs_in_care",
        "discharged"
    ]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    numeric_cols = [
        "cbp_apprehended",
        "cbp_in_care",
        "transferred_to_hhs",
        "hhs_in_care",
        "discharged"
    ]

    # clean numeric columns
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("*", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()

    # ensure daily
    df = df.asfreq("D")

    # interpolate numeric only
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

    # fill any remaining
    df[numeric_cols] = df[numeric_cols].fillna(method="bfill").fillna(method="ffill")

    return df


# ======================================================
# TIME FEATURES
# ======================================================

def add_time_features(df):
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["week"] = df.index.isocalendar().week.astype(int)
    return df


# ======================================================
# FEATURE ENGINEERING
# ======================================================

def create_forecasting_features(df):

    df["lag_1"] = df["hhs_in_care"].shift(1)
    df["lag_7"] = df["hhs_in_care"].shift(7)
    df["lag_14"] = df["hhs_in_care"].shift(14)

    df["roll_mean_7"] = df["hhs_in_care"].rolling(7).mean()
    df["roll_mean_14"] = df["hhs_in_care"].rolling(14).mean()
    df["roll_std_7"] = df["hhs_in_care"].rolling(7).std()

    df["net_pressure"] = df["transferred_to_hhs"] - df["discharged"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

def time_split(df, horizon=30):

    if len(df) <= horizon + 5:
        horizon = max(7, len(df)//3)

    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    return train, test


# ======================================================
# BASELINE MODELS
# ======================================================

def naive_forecast(train, horizon):
    last_val = train["hhs_in_care"].iloc[-1]
    return np.repeat(last_val, horizon)


def moving_average_forecast(train, horizon, window=7):
    avg = train["hhs_in_care"].rolling(window).mean().iloc[-1]
    return np.repeat(avg, horizon)


# ======================================================
# SARIMA MODEL
# ======================================================

def sarima_forecast(train, horizon):

    try:
        model = SARIMAX(
            train["hhs_in_care"],
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        result = model.fit(disp=False)

        forecast_obj = result.get_forecast(steps=horizon)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()

        return forecast, conf_int

    except:
        # fallback
        forecast = naive_forecast(train, horizon)
        return forecast, None


# ======================================================
# ML MODELS
# ======================================================

def ml_forecast(train, test, model_type="rf"):

    X_train = train.drop("hhs_in_care", axis=1)
    y_train = train["hhs_in_care"]
    X_test = test.drop("hhs_in_care", axis=1)

    # keep numeric only
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # remove any nan/inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = GradientBoostingRegressor(random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return preds


# ======================================================
# METRICS
# ======================================================

def evaluate_forecast(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2)
    }


# ======================================================
# CAPACITY RISK
# ======================================================

def add_capacity_risk(df):

    df["capacity_status"] = np.where(
        df["hhs_in_care"] > 15000, "Critical",
        np.where(df["hhs_in_care"] > 12000, "Warning", "Normal")
    )

    return df


# ======================================================
# KPI CALCULATION
# ======================================================

def calculate_kpis(df):

    avg_intake = df["transferred_to_hhs"].mean()
    avg_discharge = df["discharged"].mean()

    pressure_days = (df["net_pressure"] > 0).sum()
    breach_days = (df["capacity_status"] == "Critical").sum()

    return {
        "Avg Daily Intake": round(avg_intake, 2),
        "Avg Daily Discharge": round(avg_discharge, 2),
        "Pressure Days": int(pressure_days),
        "Capacity Breach Days": int(breach_days)
    }
