"""Validate, clean, and merge Rossmann sales and store metadata."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from utils import configure_logging, save_json

LOGGER = logging.getLogger(__name__)
TRAIN_COLUMNS = {"Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"}
STORE_COLUMNS = {"Store","StoreType","Assortment","CompetitionDistance","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2","Promo2SinceWeek","Promo2SinceYear","PromoInterval"}


def validate_schema(train: pd.DataFrame, stores: pd.DataFrame) -> None:
    """Raise a clear error when competition columns are absent."""
    missing_train = sorted(TRAIN_COLUMNS - set(train.columns))
    missing_store = sorted(STORE_COLUMNS - set(stores.columns))
    if missing_train or missing_store:
        raise ValueError(f"Schema mismatch. train missing={missing_train}; store missing={missing_store}")


def preprocess_data(train: pd.DataFrame, stores: pd.DataFrame):
    """Clean source tables, merge metadata, and return a quality report."""
    validate_schema(train, stores)
    train = train.copy(); stores = stores.copy()
    quality = {
        "train_input_rows": int(len(train)),
        "store_input_rows": int(len(stores)),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "store_duplicate_rows": int(stores.duplicated().sum()),
        "train_missing_cells": int(train.isna().sum().sum()),
        "store_missing_by_column": stores.isna().sum().astype(int).to_dict(),
    }
    train = train.drop_duplicates().copy()
    stores = stores.drop_duplicates(subset=["Store"], keep="last").copy()
    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")
    if train["Date"].isna().any():
        raise ValueError("Date parsing failed for one or more training rows")
    train["StateHoliday"] = train["StateHoliday"].astype(str).replace({"0.0":"0"})
    train["Open"] = train["Open"].fillna(1).astype(int)
    for col in ["Store","DayOfWeek","Sales","Customers","Promo","SchoolHoliday"]:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    train = train.dropna(subset=["Store","Sales"]).copy()
    train = train[(train.Sales >= 0) & train.Store.gt(0)].copy()
    stores["CompetitionDistance"] = pd.to_numeric(stores["CompetitionDistance"], errors="coerce")
    stores["CompetitionDistance"] = stores["CompetitionDistance"].fillna(stores["CompetitionDistance"].median())
    for col in ["CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2SinceWeek","Promo2SinceYear"]:
        stores[col] = pd.to_numeric(stores[col], errors="coerce").fillna(0)
    stores["PromoInterval"] = stores["PromoInterval"].fillna("None")
    stores["StoreType"] = stores["StoreType"].fillna("unknown").astype(str)
    stores["Assortment"] = stores["Assortment"].fillna("unknown").astype(str)
    merged = train.merge(stores, on="Store", how="left", validate="many_to_one")
    if merged["StoreType"].isna().any():
        raise ValueError("Some stores did not match store.csv metadata")
    quality.update({
        "output_rows": int(len(merged)),
        "closed_rows": int(merged.Open.eq(0).sum()),
        "zero_sales_rows": int(merged.Sales.eq(0).sum()),
        "date_min": str(merged.Date.min().date()),
        "date_max": str(merged.Date.max().date()),
        "stores": int(merged.Store.nunique()),
    })
    return merged.sort_values(["Date","Store"]).reset_index(drop=True), quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--stores", default="data/raw/store.csv")
    parser.add_argument("--output", default="data/processed/clean_sales.csv")
    args = parser.parse_args(); configure_logging()
    train = pd.read_csv(args.train, low_memory=False)
    stores = pd.read_csv(args.stores, low_memory=False)
    clean, quality = preprocess_data(train, stores)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    save_json(quality, "reports/data_quality.json")
    LOGGER.info("Saved %s rows to %s", f"{len(clean):,}", output)


if __name__ == "__main__":
    main()
