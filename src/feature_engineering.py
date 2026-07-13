"""Create leakage-safe calendar, competition, and promotion features."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "Sales"
CATEGORICAL_FEATURES = ["Store","DayOfWeek","StateHoliday","StoreType","Assortment"]
NUMERIC_FEATURES = [
    "Open","Promo","SchoolHoliday","CompetitionDistance","Promo2",
    "year","month","day","weekofyear","quarter","dayofyear","days_since_start",
    "month_sin","month_cos","dow_sin","dow_cos","competition_open_months",
    "promo2_active","is_month_start","is_month_end","is_weekend",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features available at forecast time; excludes Customers to prevent leakage."""
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["year"] = data.Date.dt.year
    data["month"] = data.Date.dt.month
    data["day"] = data.Date.dt.day
    data["weekofyear"] = data.Date.dt.isocalendar().week.astype(int)
    data["quarter"] = data.Date.dt.quarter
    data["dayofyear"] = data.Date.dt.dayofyear
    data["days_since_start"] = (data.Date - data.Date.min()).dt.days
    data["month_sin"] = np.sin(2*np.pi*data.month/12)
    data["month_cos"] = np.cos(2*np.pi*data.month/12)
    data["dow_sin"] = np.sin(2*np.pi*data.DayOfWeek/7)
    data["dow_cos"] = np.cos(2*np.pi*data.DayOfWeek/7)
    competition_start = data.CompetitionOpenSinceYear*12 + data.CompetitionOpenSinceMonth
    current_month = data.year*12 + data.month
    data["competition_open_months"] = np.where(data.CompetitionOpenSinceYear.gt(0), np.maximum(0, current_month-competition_start), 0)
    promo_start_year = data.Promo2SinceYear.fillna(0)
    promo_start_week = data.Promo2SinceWeek.fillna(0)
    data["promo2_active"] = ((data.Promo2.eq(1)) & ((data.year > promo_start_year) | ((data.year == promo_start_year) & (data.weekofyear >= promo_start_week)))).astype(int)
    month_names = data.Date.dt.strftime("%b")
    data["promo2_active"] = data["promo2_active"] * [int(month in str(interval)) for month, interval in zip(month_names, data.PromoInterval)]
    data["is_month_start"] = data.Date.dt.is_month_start.astype(int)
    data["is_month_end"] = data.Date.dt.is_month_end.astype(int)
    data["is_weekend"] = data.DayOfWeek.isin([6,7]).astype(int)
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(str)
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    return data


def temporal_partitions(frame: pd.DataFrame, horizon_days: int = 42):
    """Create train, validation, and test windows using two six-week holdouts."""
    data = frame.sort_values("Date").copy()
    max_date = data.Date.max()
    test_start = max_date - pd.Timedelta(days=horizon_days-1)
    validation_start = test_start - pd.Timedelta(days=horizon_days)
    train = data[data.Date < validation_start]
    validation = data[(data.Date >= validation_start) & (data.Date < test_start)]
    test = data[data.Date >= test_start]
    if min(len(train),len(validation),len(test)) == 0:
        raise ValueError("Not enough date coverage for two six-week holdouts")
    return train, validation, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/clean_sales.csv")
    parser.add_argument("--output", default="data/processed/model_features.csv")
    args = parser.parse_args()
    data = engineer_features(pd.read_csv(args.input, parse_dates=["Date"]))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f"Saved {len(data):,} feature rows to {output}")


if __name__ == "__main__":
    main()
