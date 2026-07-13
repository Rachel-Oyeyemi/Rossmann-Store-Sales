"""Train a Ridge baseline and XGBoost advanced Rossmann sales model."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, MODEL_FEATURES, TARGET, temporal_partitions
from utils import configure_logging, save_json

LOGGER = logging.getLogger(__name__)


def _open_positive(frame: pd.DataFrame) -> pd.DataFrame:
    """Train only on open stores with positive sales; closed-store predictions are forced to zero."""
    return frame[(frame.Open == 1) & (frame[TARGET] > 0)].copy()


def build_baseline() -> Pipeline:
    """Return a regularized linear baseline using one-hot categoricals."""
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=8.0))])


def build_advanced() -> Pipeline:
    """Return a nonlinear boosted-tree model with compact ordinal encoding."""
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
    ])
    model = XGBRegressor(
        n_estimators=450,
        max_depth=7,
        learning_rate=0.045,
        min_child_weight=8,
        subsample=.85,
        colsample_bytree=.9,
        reg_alpha=.05,
        reg_lambda=4.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_models(frame: pd.DataFrame):
    """Fit both models on chronological training data and return all partitions."""
    train, validation, test = temporal_partitions(frame)
    train_fit = _open_positive(train)
    X_train = train_fit[MODEL_FEATURES]
    y_train = np.log1p(train_fit[TARGET])
    baseline = build_baseline().fit(X_train, y_train)
    advanced = build_advanced().fit(X_train, y_train)
    return baseline, advanced, train, validation, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/model_features.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args(); configure_logging()
    frame = pd.read_csv(args.input, parse_dates=["Date"])
    baseline, advanced, train, validation, test = train_models(frame)
    directory = Path(args.models_dir); directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, directory / "baseline_ridge.joblib")
    joblib.dump(advanced, directory / "advanced_xgboost.joblib")
    joblib.dump({"validation": validation, "test": test}, directory / "evaluation_partitions.joblib")
    save_json({
        "target": TARGET,
        "features": MODEL_FEATURES,
        "split_strategy": "chronological train + 42-day validation + 42-day test",
        "target_transform": "log1p",
        "customers_excluded": True,
        "random_seed": 42,
        "training_rows": int(len(train)),
    }, directory / "model_metadata.json")
    LOGGER.info("Saved trained models to %s", directory)


if __name__ == "__main__":
    main()
