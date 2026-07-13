# Exploratory Data Analysis Report

## Dataset Profile
The competition training table contains 1,017,209 daily store records and nine columns. The public test table contains 41,088 rows and forecasts approximately six weeks for 1,115 stores. The target is `Sales`; `Customers` appears in training but not in the test data and is therefore excluded from modeling.

## Data Types
- Identifiers and flags: `Store`, `DayOfWeek`, `Open`, `Promo`, `SchoolHoliday`
- Date: `Date`
- Numeric outcomes: `Sales`, `Customers`
- Categorical holiday: `StateHoliday`
- Store metadata: store type, assortment, competition timing/distance, and Promo2 timing/interval

## Missing Values
The primary training table is generally complete. Store metadata contains structural missingness: competition opening dates may be unknown, while Promo2 fields are absent when a store does not participate. The competition test set is known to contain a small number of missing `Open` values. The pipeline fills missing `Open` as open and treats absent Promo2 history as no active extended promotion.

## Duplicates
The pipeline measures and removes exact duplicates from both source tables. Store metadata is constrained to one row per store before merging.

## Target Analysis
Sales are nonnegative, right-skewed, seasonal, and zero whenever stores are closed. RMSPE is computed only where actual sales are positive, matching the competition logic.

## Distribution and Outlier Analysis
High sales values are not automatically removed because they may represent genuine promotion, holiday, location, or assortment effects. The model trains on `log1p(Sales)` to reduce skew and the influence of extreme observations.

## Relationships
- Promotions generally increase average sales.
- Day of week and store type produce repeatable demand differences.
- School and state holidays change traffic patterns.
- Competition distance and duration can affect store performance.
- Calendar seasonality and December demand are material.

## Data-Quality Risks
1. `Customers` would create inference-time leakage.
2. Random splitting would leak future seasonal patterns into training.
3. Missing competition dates can mean unknown, not necessarily no competitor.
4. Promo2 interval logic must be aligned to forecast month.
5. Closed stores should not receive positive forecasts.
6. Historical competition data may not represent current retail behavior.

## Visuals
See `visuals/` for sales distribution, monthly trend, promotion lift, day-of-week effects, store-type effects, correlations, and model comparisons.
