import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ======================================================
# DATA LOADING
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

    # remove duplicate dates (IMPORTANT FIX)
    df = df.drop_duplicates(subset="date")

    df.set_index("date", inplace=True)

    numeric_cols = df.columns

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("*", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # now safe to set daily frequency
    df = df.asfreq("D")

    df[numeric_cols] = df[numeric_cols].interpolate()
    df = df.bfill().ffill()

    return df

# ======================================================
# FEATURE ENGINEERING (NO LEAKAGE)
# ======================================================

def create_forecasting_features(df):

    # Target lags
    df["lag_1"] = df["hhs_in_care"].shift(1)
    df["lag_7"] = df["hhs_in_care"].shift(7)
    df["lag_14"] = df["hhs_in_care"].shift(14)

    # Rolling from past only
    df["roll_mean_7"] = df["hhs_in_care"].shift(1).rolling(7).mean()
    df["roll_mean_14"] = df["hhs_in_care"].shift(1).rolling(14).mean()

    # External variables lagged
    df["transfer_lag1"] = df["transferred_to_hhs"].shift(1)
    df["discharge_lag1"] = df["discharged"].shift(1)

    df = df.dropna()

    return df


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

def time_split(df, horizon=30):
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    return train, test


# ======================================================
# BASELINE MODEL
# ======================================================

def naive_forecast(train, horizon):
    last_val = train["hhs_in_care"].iloc[-1]
    return np.repeat(last_val, horizon)


# ======================================================
# SARIMA MODEL
# ======================================================

def sarima_forecast(train, horizon):

    model = SARIMAX(
        train["hhs_in_care"],
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)
    forecast_obj = result.get_forecast(steps=horizon)

    return forecast_obj.predicted_mean, forecast_obj.conf_int()


# ======================================================
# RANDOM FOREST (SAFE)
# ======================================================

def ml_forecast(train, test):

    X_train = train.drop("hhs_in_care", axis=1)
    y_train = train["hhs_in_care"]
    X_test = test.drop("hhs_in_care", axis=1)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

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
# CAPACITY KPI
# ======================================================

def add_capacity_risk(df):

    df["capacity_status"] = np.where(
        df["hhs_in_care"] > 15000,
        "Critical",
        np.where(df["hhs_in_care"] > 12000, "Warning", "Normal")
    )

    return df


def calculate_kpis(df):

    df["net_pressure"] = df["transferred_to_hhs"] - df["discharged"]

    return {
        "Avg Daily Intake": round(df["transferred_to_hhs"].mean(), 2),
        "Avg Daily Discharge": round(df["discharged"].mean(), 2),
        "Pressure Days": int((df["net_pressure"] > 0).sum()),
        "Capacity Breach Days": int((df["capacity_status"] == "Critical").sum())
    }
