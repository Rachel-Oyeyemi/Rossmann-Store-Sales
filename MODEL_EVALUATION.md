# Model Evaluation

## Primary Metric — RMSPE
RMSPE measures percentage error and is the competition metric. It evaluates relative forecast quality across stores with different sales scales. Actual zero-sales rows are excluded from the denominator.

## Supporting Metrics
- RMSE: emphasizes large absolute misses
- MAE: interpretable average sales-unit error
- MAPE: average percentage error on positive actual sales
- R²: explained variance, used only as a supporting diagnostic

## Evaluation Protocol
1. Train on the earliest history.
2. Use the next six weeks for model development.
3. Evaluate once on the latest six-week holdout.
4. Report validation and test metrics for both models.
5. Slice errors by store, week, promotion, holiday, and store type before deployment.

## Business Interpretation
RMSPE should not be the only operational gate. Teams should also monitor bias, error concentration among high-volume stores, promotion-day misses, and downstream inventory or staffing outcomes.

## Bundled Demo Results
- Ridge test RMSPE: **0.115**
- XGBoost test RMSPE: **0.118**
- Ridge test MAE: **522** sales units
- XGBoost test MAE: **538** sales units

These metrics are from deterministic Rossmann-like data and validate the workflow, not competition performance.
