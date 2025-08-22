# -*- coding: utf-8 -*-
import argparse, os, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

def load_data(path):
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df.set_index("Date")
    # normalize common column names
    for cand in ["Page.Loads","page_loads","PageLoads","Page_Loads","pageLoads"]:
        if cand in df.columns:
            df = df.rename(columns={cand:"Page.Loads"})
    # clean commas if strings
    if df["Page.Loads"].dtype == object:
        df["Page.Loads"] = df["Page.Loads"].astype(str).str.replace(",","").astype(float)
    return df

def featurize(df):
    X = pd.DataFrame(index=df.index)
    X["Lag_1"] = df["Page.Loads"].shift(1)
    X["Lag_7"] = df["Page.Loads"].shift(7)
    X["RollingMean_7"] = df["Page.Loads"].shift(1).rolling(7).mean()
    X["RollingStd_7"] = df["Page.Loads"].shift(1).rolling(7).std()
    X["DayOfWeek"] = df.index.dayofweek
    X["Month"] = df.index.month
    y = df["Page.Loads"]
    feat = pd.concat([X, y], axis=1).dropna()
    return feat.iloc[:,:-1], feat.iloc[:,-1]

def model_compare(df, horizon, out):
    # Split last horizon as test
    X, y = featurize(df)
    X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    # RF
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # XGB
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    # SARIMA
    sarima = SARIMAX(y.iloc[:-horizon], order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    sarima_pred = sarima.forecast(steps=horizon)

    # Prophet
    p_df = df["Page.Loads"].reset_index().rename(columns={{"Date":"ds","Page.Loads":"y"}}) if "Date" in df.reset_index().columns else df.reset_index().rename(columns={{df.index.name or "index":"ds","Page.Loads":"y"}})
    p = Prophet(daily_seasonality=True)
    p.fit(p_df)
    future = p.make_future_dataframe(periods=horizon)
    p_fc = p.predict(future)
    prophet_pred = p_fc.set_index("ds")["yhat"].iloc[-horizon:]

    # Metrics
    def metrics(true, pred):
        rmse = float(np.sqrt(mean_squared_error(true, pred)))
        mae = float(mean_absolute_error(true, pred))
        mape = float(np.mean(np.abs((true - pred) / true)) * 100)
        return {{ "RMSE": rmse, "MAE": mae, "MAPE": mape }}

    results = {{
        "RandomForest": metrics(y_test.values, rf_pred),
        "XGBoost": metrics(y_test.values, xgb_pred),
        "SARIMA": metrics(y_test.values, sarima_pred.values),
        "Prophet": metrics(y_test.values, prophet_pred.values),
    }}

    Path(out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--do_shap", action="store_true")
    ap.add_argument("--do_optuna", action="store_true")
    args = ap.parse_args()

    df = load_data(args.data)
    results = model_compare(df, args.horizon, args.out)
    print(results)

if __name__ == "__main__":
    main()
