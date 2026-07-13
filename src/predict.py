"""Load a trained model and forecast one or more Rossmann store-days."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from feature_engineering import engineer_features, MODEL_FEATURES


def forecast(frame: pd.DataFrame, model_path: str | Path = "models/advanced_xgboost.joblib") -> pd.DataFrame:
    """Engineer features, score rows, and force closed-store sales to zero."""
    model = joblib.load(model_path)
    data = engineer_features(frame)
    predictions = np.zeros(len(data))
    open_mask = data.Open.eq(1).to_numpy()
    if open_mask.any():
        predictions[open_mask] = np.expm1(model.predict(data.loc[open_mask, MODEL_FEATURES]))
    result = frame.copy()
    result["PredictedSales"] = np.maximum(0, predictions).round(2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/advanced_xgboost.joblib")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()
    scored = forecast(pd.read_csv(args.input), args.model)
    output=Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); scored.to_csv(output,index=False)
    print(f"Saved {len(scored):,} predictions to {output}")


if __name__ == "__main__":
    main()
