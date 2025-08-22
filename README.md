# Website Traffic Forecasting

[![CI](https://github.com/asirjeril/Website_Traffic_Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/asirjeril/Website_Traffic_Forecasting/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asirjeril/Website_Traffic_Forecasting/blob/main/notebooks/Website_Traffic_Forecasting.ipynb)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red)](#streamlit-app)

Advanced time-series forecasting of daily website traffic with **ARIMA/SARIMA**, **Prophet**, **Random Forest**, **XGBoost**, and **LSTM**, plus **Optuna** tuning and **SHAP** explainability. Includes a one-click **Streamlit** app.

---

## 🗂️ Repository Structure
```
.
├── .github/workflows/ci.yml               # Lint + tests on push/PR
├── app/app.py                             # Streamlit app entrypoint
├── data/
│   ├── daily-website-visitors.csv
│   └── sample.csv                         # tiny sample for CI / examples
├── docs/
│   ├── model_card.md
│   └── project_description.md
├── notebooks/
│   └── Website_Traffic_Forecasting.ipynb  # E2E exploration & modeling
├── reports/
│   └── figures/                           # export plots here
├── src/traffic_forecasting/
│   ├── __init__.py
│   └── train_and_forecast.py              # CLI for training & forecasting
├── tests/
│   └── test_smoke.py
├── .gitignore
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### CLI: Train & Forecast
```bash
python -m src.traffic_forecasting.train_and_forecast   --data data/daily-website-visitors.csv   --horizon 30   --out reports/figures   --do_shap --do_optuna
```

### Streamlit App
```bash
streamlit run app/app.py
```
Upload a CSV like `data/daily-website-visitors.csv` and interactively forecast the next 30 days.

## 📓 Notebook
Open `notebooks/Website_Traffic_Forecasting.ipynb` or the **Open in Colab** badge above.

## 🔍 Methods (high level)
- Stationarity checks (ADF), decomposition, ACF/PACF
- ARIMA/SARIMA, Prophet
- Feature-engineered ML: RF / XGBoost (lags, rolling stats, calendar features)
- LSTM baseline
- **Optuna** hyperparameter tuning
- **SHAP** global feature importance (for tree models)
- Model comparison via RMSE/MAE/MAPE and final plot

## 📄 Model Card
See `docs/model_card.md` for intended use, metrics, data assumptions, risks, and maintenance.

---

## 📜 License
MIT © 2025 asirjeril
