# Model Comparison

## Baseline — Ridge Regression
Ridge provides a transparent and reproducible benchmark. One-hot encoded categorical variables and standardized numeric features capture additive store, calendar, promotion, holiday, and competition effects. It is fast, stable, and useful for diagnosing whether nonlinear complexity adds value.

## Advanced — XGBoost
XGBoost captures interactions and nonlinearities such as promotion effects varying by store type, seasonal patterns varying by assortment, and competition effects changing over time. Regularization, row subsampling, and column subsampling reduce overfitting.

## Validation Design
Rows are sorted by date. The latest 42 days form the untouched test period and the preceding 42 days form validation. The model trains on all earlier history. This mirrors the competition horizon and avoids future-to-past leakage.

## Target and Inference Rules
- Train on `log1p(Sales)` for open stores with positive sales.
- Exclude `Customers` because it is unavailable in the competition test set.
- Force predictions to zero whenever `Open = 0`.

## Recommendation
Use XGBoost when it materially improves test RMSPE over Ridge and remains stable across stores, promotion status, and forecast weeks. Keep Ridge as the interpretable champion–challenger baseline.

## Bundled Demo Benchmark
| Model | Validation RMSPE | Test RMSPE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|
| Ridge Regression | 0.113 | 0.115 | 738 | 522 | 0.937 |
| XGBoost | 0.117 | 0.118 | 789 | 538 | 0.928 |

Ridge won the portable demonstration on RMSPE. This result is reported honestly rather than selecting the more complex model by default. The official competition data may produce a different ranking.
