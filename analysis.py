#IMPORT LIBRARIES

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# LOAD DATA AND PREPARE DATA

def load_data(path="data/HHS_Unaccompanied_Alien_Children_Program.csv"):
    df = pd.read_csv(path)

    # rename columns
    df.columns = [
        "Date",
        "cbp_apprehended",
        "cbp_in_care",
        "transferred_to_hhs",
        "hhs_in_care",
        "discharged"
    ]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    # ensure daily continuity
    df = df.asfreq("D")
    df.interpolate(method="linear", inplace=True)

    return df


# FEATURE ENGINEERING

def create_features(df):

    df["lag1"] = df["hhs_in_care"].shift(1)
    df["lag7"] = df["hhs_in_care"].shift(7)
    df["lag14"] = df["hhs_in_care"].shift(14)

    df["roll7"] = df["hhs_in_care"].rolling(7).mean()
    df["roll14"] = df["hhs_in_care"].rolling(14).mean()

    df["net_pressure"] = df["transferred_to_hhs"] - df["discharged"]

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df.dropna(inplace=True)
    return df


# SPLIT

def split_data(df, horizon=30):
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    return train, test


# ARIMA FORECAST

def arima_forecast(train, horizon):

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


# ML MODELS

def ml_model(train, test, model_type="rf"):

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


# BASELINE

def naive_model(train, test):
    last = train["hhs_in_care"].iloc[-1]
    return np.repeat(last, len(test))


# METRICS

def metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred)/actual))*100

    return round(mae,2), round(rmse,2), round(mape,2)


# CAPACITY RISK

def risk_flag(df):

    df["capacity_status"] = np.where(
        df["hhs_in_care"] > 15000, "Critical",
        np.where(df["hhs_in_care"] > 12000, "Warning", "Normal")
    )

    return df
