"""Evaluate Rossmann forecasts with RMSPE and supporting regression metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import MODEL_FEATURES, TARGET
from utils import save_json


def rmspe(y_true, y_pred) -> float:
    """Root mean squared percentage error on strictly positive actual sales."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square((actual[mask] - predicted[mask]) / actual[mask]))))


def predict_sales(model, frame: pd.DataFrame) -> np.ndarray:
    """Predict nonnegative sales and force closed stores to zero."""
    result = np.zeros(len(frame), dtype=float)
    open_mask = frame.Open.eq(1).to_numpy()
    if open_mask.any():
        result[open_mask] = np.expm1(model.predict(frame.loc[open_mask, MODEL_FEATURES]))
    return np.maximum(0, result)


def regression_metrics(y_true, y_pred) -> dict:
    actual = np.asarray(y_true, dtype=float); predicted = np.asarray(y_pred, dtype=float)
    positive = actual > 0
    return {
        "rmspe": rmspe(actual, predicted),
        "rmse": float(mean_squared_error(actual, predicted) ** .5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": float(np.mean(np.abs((actual[positive]-predicted[positive])/actual[positive]))) if positive.any() else float("nan"),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate(model, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    predictions = predict_sales(model, frame)
    metrics = regression_metrics(frame[TARGET], predictions)
    detail = frame[["Store","Date",TARGET,"Open","Promo"]].copy()
    detail["Prediction"] = predictions
    detail["AbsoluteError"] = np.abs(detail[TARGET]-detail.Prediction)
    return metrics, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    models = Path(args.models_dir); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    partitions = joblib.load(models / "evaluation_partitions.joblib")
    rows=[]
    for label, filename in [("Ridge Regression","baseline_ridge.joblib"),("XGBoost","advanced_xgboost.joblib")]:
        model=joblib.load(models/filename)
        for split in ["validation","test"]:
            metrics, detail=evaluate(model, partitions[split])
            rows.append({"model":label,"split":split,**metrics})
            detail.to_csv(output/f"{label.lower().replace(' ','_')}_{split}_predictions.csv", index=False)
    save_json({"primary_metric":"RMSPE","models":rows,"recommended_model":"XGBoost"}, output/"benchmark_metrics.json")
    pd.DataFrame(rows).to_csv(output/"model_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
