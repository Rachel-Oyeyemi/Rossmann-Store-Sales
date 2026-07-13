"""Streamlit application for Rossmann sales forecasting."""
from __future__ import annotations
import json,sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from generate_demo_data import generate_demo
from preprocess import preprocess_data
from feature_engineering import engineer_features
from train_model import train_models
from evaluate_model import predict_sales

st.set_page_config(page_title="Rossmann Sales Forecasting",page_icon="🏪",layout="wide")

@st.cache_resource
def load_model():
    path=ROOT/"models/advanced_xgboost.joblib"
    if path.exists(): return joblib.load(path),"Committed model artifact"
    raw,stores=generate_demo(n_stores=25,start="2014-01-01",end="2015-07-31")
    clean,_=preprocess_data(raw,stores); features=engineer_features(clean); _,model,_,_,_=train_models(features)
    return model,"Cached deterministic fallback"

@st.cache_data
def load_json(path): return json.loads(path.read_text()) if path.exists() else {}

MODEL,SCOPE=load_model(); METRICS=load_json(ROOT/"reports/benchmark_metrics.json")
page=st.sidebar.radio("Navigate",["Home","Project Overview","Prediction Interface","Model Performance","Visualizations","Business Insights","About"])
st.sidebar.caption(SCOPE); st.sidebar.warning("Portfolio demonstration. Retrain on authenticated Kaggle files before operational use.")

if page=="Home":
    st.title("Rossmann Store Sales Forecasting")
    c=st.columns(4); c[0].metric("Training rows","1,017,209"); c[1].metric("Stores","1,115"); c[2].metric("Forecast horizon","6 weeks"); c[3].metric("Primary metric","RMSPE")
    st.markdown("A leakage-safe daily sales forecasting project using calendar, promotion, competition, holiday, and store metadata.")
elif page=="Project Overview":
    st.header("Project Overview")
    st.markdown("""**Business question:** How much will each store sell over the next six weeks so teams can plan inventory, staffing, promotions, and cash flow?

**Validation:** chronological six-week windows; no random split. `Customers` is excluded because it is unavailable in the competition test set.""")
elif page=="Prediction Interface":
    st.header("Single Store-Day Forecast")
    with st.form("forecast"):
        c1,c2,c3=st.columns(3)
        with c1:
            store=st.number_input("Store",1,1115,1); date=st.date_input("Date",pd.Timestamp("2015-08-01")); dow=pd.Timestamp(date).dayofweek+1; open_value=st.selectbox("Open",[1,0]); promo=st.selectbox("Promo",[0,1])
        with c2:
            state=st.selectbox("State Holiday",["0","a","b","c"]); school=st.selectbox("School Holiday",[0,1]); store_type=st.selectbox("Store Type",list("abcd")); assortment=st.selectbox("Assortment",list("abc"))
        with c3:
            distance=st.number_input("Competition Distance",0.0,100000.0,1000.0); promo2=st.selectbox("Promo2",[0,1]); comp_month=st.number_input("Competition Open Month",0,12,1); comp_year=st.number_input("Competition Open Year",0,2026,2010)
        submitted=st.form_submit_button("Forecast Sales",use_container_width=True)
    if submitted:
        row=pd.DataFrame([{"Store":store,"DayOfWeek":dow,"Date":pd.Timestamp(date),"Sales":0,"Customers":0,"Open":open_value,"Promo":promo,"StateHoliday":state,"SchoolHoliday":school,"StoreType":store_type,"Assortment":assortment,"CompetitionDistance":distance,"CompetitionOpenSinceMonth":comp_month,"CompetitionOpenSinceYear":comp_year,"Promo2":promo2,"Promo2SinceWeek":1,"Promo2SinceYear":2014,"PromoInterval":"Jan,Apr,Jul,Oct"}])
        featured=engineer_features(row); prediction=predict_sales(MODEL,featured)[0]
        st.metric("Predicted daily sales",f"{prediction:,.0f}")
elif page=="Model Performance":
    st.header("Model Performance")
    if METRICS: st.dataframe(pd.DataFrame(METRICS.get("models",[])),use_container_width=True); st.markdown(f"**Recommended model:** {METRICS.get('recommended_model','—')}"); st.caption(METRICS.get("benchmark_scope",""))
    for name in ["model_comparison.svg","error_metrics.svg"]:
        path=ROOT/"visuals"/name
        if path.exists(): st.image(str(path),use_container_width=True)
elif page=="Visualizations":
    st.header("Exploratory Visualizations")
    for name in ["sales_distribution.svg","monthly_sales.svg","promo_lift.svg","sales_by_day.svg","sales_by_store_type.svg","correlation_heatmap.svg"]:
        path=ROOT/"visuals"/name
        if path.exists(): st.image(str(path),caption=name.replace("_"," ").replace(".svg","").title(),use_container_width=True)
elif page=="Business Insights":
    st.header("Business Insights")
    st.markdown("""1. Promotions create material lift but require incremental-profit testing.
2. Store and day-of-week effects justify store-level forecasts.
3. Closed stores must be forced to zero.
4. Forecast error should be tracked by store, horizon, promotion status, and season.
5. Forecasts should feed inventory and staffing decisions with uncertainty buffers.""")
else:
    st.header("About")
    st.markdown("Built by **Rachel Oyeyemi** as a recruiter-ready Data Analytics & AI portfolio project. It demonstrates forecasting, leakage prevention, feature engineering, gradient boosting, reproducibility, testing, Streamlit, and executive communication.")
