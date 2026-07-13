"""Run the complete Rossmann data, modeling, evaluation, and visualization workflow."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from generate_demo_data import generate_demo
from preprocess import preprocess_data
from feature_engineering import engineer_features
from train_model import train_models
from evaluate_model import evaluate
from visualize import generate_visuals
from utils import configure_logging,save_json


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",default="data/raw/train.csv")
    parser.add_argument("--stores",default="data/raw/store.csv")
    args=parser.parse_args(); configure_logging()
    train_path=Path(args.train); store_path=Path(args.stores)
    if train_path.exists() and store_path.exists():
        raw_train=pd.read_csv(train_path,low_memory=False); raw_store=pd.read_csv(store_path,low_memory=False); scope="Official Kaggle competition files"
    else:
        raw_train,raw_store=generate_demo(); scope="Deterministic Rossmann-like demo data"
    clean,quality=preprocess_data(raw_train,raw_store)
    features=engineer_features(clean)
    (ROOT/"data/processed").mkdir(parents=True,exist_ok=True)
    clean.to_csv(ROOT/"data/processed/clean_sales.csv",index=False)
    features.to_csv(ROOT/"data/processed/model_features.csv",index=False)
    baseline,advanced,train,validation,test=train_models(features)
    (ROOT/"models").mkdir(exist_ok=True)
    joblib.dump(baseline,ROOT/"models/baseline_ridge.joblib")
    joblib.dump(advanced,ROOT/"models/advanced_xgboost.joblib")
    joblib.dump({"validation":validation,"test":test},ROOT/"models/evaluation_partitions.joblib")
    (ROOT/"reports").mkdir(exist_ok=True)
    rows=[]
    for label,model in [("Ridge Regression",baseline),("XGBoost",advanced)]:
        for split_name,partition in [("validation",validation),("test",test)]:
            metrics,detail=evaluate(model,partition); rows.append({"model":label,"split":split_name,**metrics})
            detail.to_csv(ROOT/f"reports/{label.lower().replace(' ','_')}_{split_name}_predictions.csv",index=False)
    pd.DataFrame(rows).to_csv(ROOT/"reports/model_metrics.csv",index=False)
    recommended=min((r for r in rows if r["split"]=="test"),key=lambda r:r["rmspe"])["model"]
    save_json({"benchmark_scope":scope,"primary_metric":"RMSPE","models":rows,"recommended_model":recommended,"quality":quality},ROOT/"reports/benchmark_metrics.json")
    save_json(quality,ROOT/"reports/data_quality.json")
    generate_visuals(clean,features)
    print(pd.DataFrame(rows).to_string(index=False)); print(f"Recommended model: {recommended}")


if __name__=="__main__": main()
