import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================
# LOAD DATA
# =========================
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
    df = df.sort_values("date").drop_duplicates(subset="date")
    df.set_index("date", inplace=True)

    # clean numbers
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("*", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # continuous dates
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)

    df = df.interpolate().bfill().ffill()

    return df


# =========================
# FEATURES
# =========================
def create_forecasting_features(df):

    # lag features
    df["lag_1"] = df["hhs_in_care"].shift(1)
    df["lag_7"] = df["hhs_in_care"].shift(7)
    df["lag_14"] = df["hhs_in_care"].shift(14)

    # rolling
    df["roll_mean_7"] = df["hhs_in_care"].rolling(7).mean()
    df["roll_mean_14"] = df["hhs_in_care"].rolling(14).mean()
    df["roll_std_7"] = df["hhs_in_care"].rolling(7).std()

    # VERY IMPORTANT
    df["net_pressure"] = df["transferred_to_hhs"] - df["discharged"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


# =========================
# SPLIT
# =========================
def time_split(df, horizon=30):
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    return train, test


# =========================
# MODELS
# =========================
def naive_forecast(train, horizon):
    return np.repeat(train["hhs_in_care"].iloc[-1], horizon)


def sarima_forecast(train, horizon):

    try:
        model = SARIMAX(
            train["hhs_in_care"],
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        res = model.fit(disp=False)
        fc = res.get_forecast(steps=horizon)

        return fc.predicted_mean, fc.conf_int()

    except:
        return naive_forecast(train, horizon), None


def ml_forecast(train, test):

    X_train = train.drop("hhs_in_care", axis=1)
    y_train = train["hhs_in_care"]
    X_test = test.drop("hhs_in_care", axis=1)

    X_train = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_test = X_test.select_dtypes(include=[np.number]).fillna(0)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model.predict(X_test)


# =========================
# METRICS
# =========================
def evaluate_forecast(actual, pred):

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100

    return {
        "MAE": round(mae,2),
        "RMSE": round(rmse,2),
        "MAPE (%)": round(mape,2)
    }


# =========================
# CAPACITY + KPI
# =========================
def add_capacity_risk(df):

    df["capacity_status"] = np.where(
        df["hhs_in_care"] > 15000,"Critical",
        np.where(df["hhs_in_care"] > 12000,"Warning","Normal")
    )
    return df


def calculate_kpis(df):

    # safety check
    if "net_pressure" not in df.columns:
        df["net_pressure"] = df["transferred_to_hhs"] - df["discharged"]

    if "capacity_status" not in df.columns:
        df = add_capacity_risk(df)

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
