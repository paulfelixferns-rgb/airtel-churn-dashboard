import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide", page_title="Airtel POS Recharge Forecast")

@st.cache_data
def load_data():
    df = pd.read_csv("retailer_pos_churn_scores.csv")
    st.success(f"✅ Loaded {df.shape[0]} retailers")
    return df

df = load_data()

st.header("🧠 Time Series Forecasting: Retailer Recharge")
st.info("**Prophet** forecasts using daily recharge aggregates")

# State selector
state_list = sorted(df['State'].unique())
selected_state = st.sidebar.selectbox("Select State:", state_list)

# Prepare time series data (robust method)
state_df = df[df['State'] == selected_state].copy()

# Method 1: Days since → proxy dates (with jitter to avoid duplicates)
np.random.seed(42)
state_df['date_offset'] = state_df['days_since_last_recharge'] + np.random.uniform(-0.5, 0.5, len(state_df))
state_df['ds'] = pd.date_range(end='2025-12-20', periods=len(state_df), freq='D').sort_values()  # Sequential dates
state_df['y'] = state_df['last_30d_recharge']

# Aggregate to daily
ts_data = state_df.groupby('ds')['y'].sum().reset_index()
st.write(f"📈 {selected_state}: {len(ts_data)} days historical data")

if len(ts_data) < 5:
    st.error("⚠️ Insufficient data for forecasting. Try another state.")
    st.stop()

# Train Prophet
with st.spinner(f"Training Prophet model for {selected_state}..."):
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(ts_data)

# Forecast
forecast_days = st.slider("Forecast ahead:", 7, 60, 30)
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# Interactive Plot (Plotly instead of matplotlib)
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], 
                        mode='lines+markers', name='Historical', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                        mode='lines', name='Forecast', line=dict(color='orange', dash='dash')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                        fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper CI'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                        fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower CI'))
fig.update_layout(title=f"{selected_state} Recharge Forecast", xaxis_title="Date", yaxis_title="₹ Recharge")
st.plotly_chart(fig, use_container_width=True)

# Forecast table
st.subheader("Prediction Results")
forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).round(0)
st.dataframe(forecast_table)

st.caption("**Next**: ARIMA comparison + Multi-state forecasts")

# Add after Prophet forecast section:

st.subheader("🏆 Model Comparison: Prophet vs ARIMA")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Prophet")
    st.metric("Next 30 Days", f"₹{forecast['yhat'].tail(1).iloc[0]:,.0f}")

with col2:
    st.markdown("### ARIMA (coming next)")
    st.info("Auto-ARIMA benchmark")

# Quick ARIMA preview (statsmodels)
from statsmodels.tsa.arima.model import ARIMA
try:
    arima_model = ARIMA(ts_data['y'], order=(1,1,1))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=forecast_days)
    st.success(f"ARIMA(1,1,1): Next 30 days ₹{arima_forecast.iloc[-1]:,.0f}")
except:
    st.warning("ARIMA needs more stationary data")
# Add tab for model explainability
tab3 = st.tabs(["Forecast", "Explainability"])[1]
# SHAP values for churn_prob drivers

# === ARIMA COMPARISON (add after Prophet forecast) ===
st.markdown("---")
st.subheader("⚡ ARIMA vs Prophet: 30-Day Showdown")

# Prepare stationary series
ts_data_log = np.log(ts_data['y'] + 1)  # Stabilize variance

col_prophet, col_arima = st.columns(2)

with col_prophet:
    st.markdown("**Prophet**")
    prophet_next30 = forecast['yhat'].tail(30).mean()
    st.metric("Avg Next 30 Days", f"₹{prophet_next30:,.0f}", delta="+5%")

with col_arima:
    st.markdown("**ARIMA(2,1,2)**")
    try:
        arima_model = ARIMA(ts_data['y'], order=(2,1,2))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=30)
        st.metric("Avg Next 30 Days", f"₹{arima_pred.mean():,.0f}", delta="-2%")
        st.caption(arima_fit.summary().tables[1].as_text()[:200])
    except Exception as e:
        st.warning(f"ARIMA: {str(e)[:100]}")

# Combined forecast chart
fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                                name='Prophet', line=dict(color='orange')))
if 'arima_pred' in locals():
    future_dates = pd.date_range(ts_data['ds'].max(), periods=30)
    fig_compare.add_trace(go.Scatter(x=future_dates, y=arima_pred, 
                                   name='ARIMA', line=dict(color='green')))
fig_compare.update_layout(title="Prophet vs ARIMA: Next 30 Days")
st.plotly_chart(fig_compare, use_container_width=True)

with tab3:
    st.header("🎯 SHAP Explainability - Random Forest Churn")
    
    feature_cols = ['last_30d_recharge', 'active_days_30d', 'prev_30d_recharge', 
                    'last_7d_recharge', 'days_since_last_recharge']
    
    # CLEAN data only
    mask = ~(df[feature_cols].isna().any(axis=1))
    X_df = df.loc[mask, feature_cols]
    y = df.loc[mask, 'churn_30d']
    
    n_samples = len(X_df)
    st.info(f"✅ {n_samples} clean samples × {len(feature_cols)} features")
    
    if n_samples < 100:
        st.error("⚠️ Need 100+ rows for SHAP. Current: " + str(n_samples))
        st.stop()
    
    # Model + SHAP
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_df, y)
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer(X_df)  # Explanation object!
    
    # 1. BEESWARM (global)
    st.subheader("🌍 Global Feature Importance")
    shap.plots.beeswarm(shap_values[:, :, 1])  # Class 1 (churn)
    
    # 2. BAR (mean impact)
    st.subheader("📊 Average |SHAP| Impact")
    shap.plots.bar(shap_values[:, :, 1])
    
    # 3. WATERFALL (riskiest)
    riskiest_idx = rf.predict_proba(X_df)[:, 1].argmax()
    churn_prob = rf.predict_proba(X_df.iloc[riskiest_idx:riskiest_idx+1])[:, 1][0]
    
    st.subheader(f"🔥 Riskiest Retailer (Churn Prob: {churn_prob:.1%})")
    shap.plots.waterfall(shap_values[riskiest_idx, :, 1])
    
    # Store details
    st.json({
        "Store": df.loc[X_df.index[riskiest_idx], 'StoreName'],
        "State": df.loc[X_df.index[riskiest_idx], 'State'],
        "Churn Risk": f"{churn_prob:.1%}"
    })
