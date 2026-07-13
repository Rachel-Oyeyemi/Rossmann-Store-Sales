"""Generate compact EDA and model-evaluation visualizations."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(name: str) -> None:
    Path("visuals").mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(Path("visuals")/name, format="svg", bbox_inches="tight"); plt.close()


def generate_visuals(clean: pd.DataFrame, featured: pd.DataFrame) -> None:
    open_sales=clean[(clean.Open==1)&(clean.Sales>0)]
    plt.figure(figsize=(8,4.5)); plt.hist(open_sales.Sales, bins=50); plt.title("Distribution of Open-Store Daily Sales"); plt.xlabel("Sales"); plt.ylabel("Store-days"); _save("sales_distribution.svg")
    monthly=clean.groupby(clean.Date.dt.to_period("M")).Sales.sum(); plt.figure(figsize=(9,4.5)); plt.plot(monthly.index.astype(str),monthly.values); plt.xticks(rotation=60); plt.title("Monthly Sales Trend"); plt.ylabel("Sales"); _save("monthly_sales.svg")
    promo=open_sales.groupby("Promo").Sales.mean(); plt.figure(figsize=(6,4)); plt.bar(["No Promo","Promo"],promo.reindex([0,1]).values); plt.title("Average Sales by Promotion Status"); plt.ylabel("Average Sales"); _save("promo_lift.svg")
    dow=open_sales.groupby("DayOfWeek").Sales.mean(); plt.figure(figsize=(7,4)); plt.bar(dow.index.astype(str),dow.values); plt.title("Average Sales by Day of Week"); plt.xlabel("Day of Week"); _save("sales_by_day.svg")
    st=open_sales.groupby("StoreType").Sales.mean().sort_values(); plt.figure(figsize=(6,4)); plt.bar(st.index,st.values); plt.title("Average Sales by Store Type"); _save("sales_by_store_type.svg")
    corr=featured[["Sales","Promo","SchoolHoliday","CompetitionDistance","month","DayOfWeek"]].corr(); plt.figure(figsize=(6,5)); plt.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1); plt.xticks(range(len(corr)),corr.columns,rotation=45,ha="right"); plt.yticks(range(len(corr)),corr.columns); plt.colorbar(); plt.title("Selected Feature Correlations"); _save("correlation_heatmap.svg")
    metrics_path=Path("reports/model_metrics.csv")
    if metrics_path.exists():
        m=pd.read_csv(metrics_path); test=m[m.split=="test"].set_index("model")
        plt.figure(figsize=(7,4)); plt.bar(test.index,test.rmspe); plt.title("Test RMSPE — Lower Is Better"); plt.ylabel("RMSPE"); _save("model_comparison.svg")
        plt.figure(figsize=(7,4)); x=np.arange(len(test)); w=.25; plt.bar(x-w,test.mae,w,label="MAE"); plt.bar(x,test.rmse,w,label="RMSE"); plt.xticks(x,test.index); plt.title("Test Error Metrics"); plt.legend(); _save("error_metrics.svg")


if __name__ == "__main__":
    clean=pd.read_csv("data/processed/clean_sales.csv",parse_dates=["Date"]); featured=pd.read_csv("data/processed/model_features.csv",parse_dates=["Date"]); generate_visuals(clean,featured)
