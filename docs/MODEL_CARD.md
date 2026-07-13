# Model Card

## Intended Use
Educational and portfolio demonstration of store-level daily sales forecasting.

## Not Intended For
Direct inventory ordering, workforce scheduling, financial reporting, or modern Rossmann operational decisions without retraining and governance.

## Inputs
Store ID, calendar features, promotion flags, holidays, store type, assortment, and competition/Promo2 metadata.

## Output
Nonnegative predicted daily sales; zero when a store is closed.

## Evaluation
Chronological six-week validation and test windows with RMSPE as the primary metric.

## Limitations
Historical competition data, no live inventory or local events, missing metadata, point forecasts without uncertainty, and no causal interpretation of promotions.
