# Technical Architecture

```text
Kaggle Competition API
        │
        ▼
data/raw/{train,test,store}.csv
        │ schema checks + deduplication
        ▼
preprocess.py ──► reports/data_quality.json
        │ merge store metadata
        ▼
feature_engineering.py
        │ leakage-safe calendar/promo/competition features
        ▼
chronological 6-week validation and test windows
        ├── Ridge baseline
        └── XGBoost advanced
        │
        ▼
evaluate_model.py ──► RMSPE / RMSE / MAE / MAPE / R²
        │
        ├── models/
        ├── reports/
        ├── visuals/
        ├── Streamlit app
        └── executive presentation
```

## Design Decisions
- No random split
- No `Customers` feature
- Zero output for closed stores
- Log-transformed target
- Reproducible random seeds
- Demo generator for portable execution
