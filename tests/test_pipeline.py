import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from generate_demo_data import generate_demo
from preprocess import preprocess_data
from feature_engineering import engineer_features,temporal_partitions,MODEL_FEATURES
from evaluate_model import rmspe,regression_metrics


def test_demo_schema_and_preprocess():
    train,stores=generate_demo(n_stores=5,start="2015-01-01",end="2015-05-31")
    clean,quality=preprocess_data(train,stores)
    assert len(clean)==len(train)
    assert quality["stores"]==5
    assert clean.StoreType.notna().all()


def test_feature_engineering_excludes_customers():
    train,stores=generate_demo(n_stores=5,start="2015-01-01",end="2015-05-31")
    clean,_=preprocess_data(train,stores); features=engineer_features(clean)
    assert "Customers" not in MODEL_FEATURES
    assert set(MODEL_FEATURES).issubset(features.columns)


def test_temporal_partitions_are_ordered():
    train,stores=generate_demo(n_stores=3,start="2014-01-01",end="2015-07-31")
    features=engineer_features(preprocess_data(train,stores)[0]); a,b,c=temporal_partitions(features)
    assert a.Date.max()<b.Date.min()<c.Date.min()


def test_rmspe_and_metrics():
    actual=np.array([100.,200.,0.]); predicted=np.array([90.,220.,5.])
    assert np.isclose(rmspe(actual,predicted),.1)
    metrics=regression_metrics(actual,predicted)
    assert set(["rmspe","rmse","mae","mape","r2"]).issubset(metrics)
