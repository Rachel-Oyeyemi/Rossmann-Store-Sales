"""Generate deterministic Rossmann-like data for portable demonstrations."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo(n_stores: int = 80, start: str = "2014-01-01", end: str = "2015-07-31", seed: int = 42):
    """Return synthetic train and store tables that mirror competition schemas."""
    rng = np.random.default_rng(seed)
    stores = pd.DataFrame({
        "Store": np.arange(1, n_stores + 1),
        "StoreType": rng.choice(list("abcd"), n_stores, p=[.55,.12,.27,.06]),
        "Assortment": rng.choice(list("abc"), n_stores, p=[.48,.08,.44]),
        "CompetitionDistance": np.maximum(50, rng.lognormal(7.3, 1.0, n_stores)).round(),
        "CompetitionOpenSinceMonth": rng.integers(1, 13, n_stores),
        "CompetitionOpenSinceYear": rng.integers(1995, 2015, n_stores),
        "Promo2": rng.binomial(1, .55, n_stores),
        "Promo2SinceWeek": rng.integers(1, 53, n_stores),
        "Promo2SinceYear": rng.integers(2009, 2015, n_stores),
        "PromoInterval": rng.choice(["Jan,Apr,Jul,Oct","Feb,May,Aug,Nov","Mar,Jun,Sept,Dec"], n_stores),
    })
    # Introduce realistic metadata gaps.
    gap = rng.choice(stores.index, max(2, n_stores // 10), replace=False)
    stores.loc[gap, ["CompetitionOpenSinceMonth","CompetitionOpenSinceYear"]] = np.nan
    no_promo = stores["Promo2"].eq(0)
    stores.loc[no_promo, ["Promo2SinceWeek","Promo2SinceYear","PromoInterval"]] = np.nan

    dates = pd.date_range(start, end, freq="D")
    grid = pd.MultiIndex.from_product([stores.Store, dates], names=["Store","Date"]).to_frame(index=False)
    grid["DayOfWeek"] = grid.Date.dt.dayofweek + 1
    grid["Open"] = ((grid.DayOfWeek != 7) | (grid.Store % 9 == 0)).astype(int)
    grid["Promo"] = (((grid.Date.dt.day <= 10) | (grid.Date.dt.day >= 25)) & (grid.DayOfWeek <= 6)).astype(int)
    grid["StateHoliday"] = "0"
    grid.loc[(grid.Date.dt.month == 1) & (grid.Date.dt.day == 1), "StateHoliday"] = "a"
    grid.loc[(grid.Date.dt.month == 12) & (grid.Date.dt.day.isin([25,26])), "StateHoliday"] = "c"
    grid["SchoolHoliday"] = grid.Date.dt.month.isin([7,8,12]).astype(int)

    store_base = rng.normal(5200, 1000, n_stores)
    store_map = dict(zip(stores.Store, store_base))
    type_mult = stores.set_index("Store").StoreType.map({"a":1.0,"b":1.22,"c":.93,"d":1.08}).to_dict()
    assortment_mult = stores.set_index("Store").Assortment.map({"a":.95,"b":1.02,"c":1.08}).to_dict()
    base = grid.Store.map(store_map).to_numpy()
    dow_effect = grid.DayOfWeek.map({1:1.12,2:1.03,3:1.00,4:1.01,5:1.08,6:.92,7:.78}).to_numpy()
    season = 1 + .12*np.sin(2*np.pi*(grid.Date.dt.dayofyear.to_numpy()/365.25)) + .08*(grid.Date.dt.month.to_numpy()==12)
    trend = 1 + .00018*(grid.Date - dates.min()).dt.days.to_numpy()
    store_type = grid.Store.map(type_mult).to_numpy()
    assortment = grid.Store.map(assortment_mult).to_numpy()
    promo_strength = (
        .13
        + .17 * (store_type > 1.15)
        + .08 * (assortment > 1.05)
        + .09 * (grid.Date.dt.month.to_numpy() == 12)
        + .06 * (grid.DayOfWeek.to_numpy() == 1)
    )
    promo = 1 + grid.Promo.to_numpy() * promo_strength
    holiday = np.where(grid.StateHoliday.ne("0"), np.where(store_type > 1.15, .72, .50), 1.0)
    school = 1 + .025*grid.SchoolHoliday.to_numpy() + .045*(grid.SchoolHoliday.to_numpy() * (assortment > 1.05))
    distance = grid.Store.map(stores.set_index("Store").CompetitionDistance.to_dict()).to_numpy()
    competition = 1 - .07*np.exp(-distance/1200)
    noise = rng.lognormal(mean=0, sigma=.105, size=len(grid))
    sales = base*dow_effect*season*trend*promo*holiday*school*store_type*assortment*competition*noise*grid.Open.to_numpy()
    sales = np.maximum(0, sales).round().astype(int)
    customers = np.where(grid.Open.eq(1), np.maximum(1, (sales / rng.normal(8.5, .7, len(grid))).round()), 0).astype(int)
    grid["Sales"] = sales
    grid["Customers"] = customers
    train = grid[["Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"]]
    return train, stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/sample_data")
    args = parser.parse_args()
    train, stores = generate_demo()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "demo_train.csv", index=False)
    stores.to_csv(output / "demo_store.csv", index=False)
    print(f"Saved {len(train):,} demo rows and {len(stores)} stores to {output}")


if __name__ == "__main__":
    main()
