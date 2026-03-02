# Predictive Forecasting of Care Load & Placement Demand

**Overview**
This project builds a predictive analytics dashboard to forecast future care load and discharge demand using time-series and machine learning models.

**Features**
Time-series forecasting using SARIMA and Random Forest

Feature engineering with lag and rolling trends

Forecast accuracy evaluation (MAE, RMSE, MAPE)

Capacity risk monitoring

Interactive Streamlit dashboard

**Tech Stack**
Python, Pandas, Scikit-learn, Statsmodels, Streamlit, Plotly

**Dataset**
Daily operational dataset containing intake, care load, and discharge metrics.

**How to Run**
pip install -r requirements.txt 
streamlit run app.py

**Project Structure**
app.py – Streamlit dashboard

analysis.py – Data processing and models

data/ – Dataset

requirements.txt – Dependencies

**Use Case**
Helps healthcare and government agencies forecast demand and plan resources proactively using predictive analytics.
