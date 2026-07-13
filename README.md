# Rossmann Store Sales Forecasting

[![CI](https://github.com/Rachel-Oyeyemi/Rossmann-Store-Sales/actions/workflows/ci.yml/badge.svg)](https://github.com/Rachel-Oyeyemi/Rossmann-Store-Sales/actions)

A recruiter-ready machine-learning portfolio project that forecasts daily sales six weeks ahead using leakage-safe feature engineering, chronological validation, Ridge Regression, XGBoost, visualization, automated tests, Streamlit, and executive communication.

> **Data note:** Kaggle competition files require authentication and acceptance of the competition rules. The repository includes a deterministic Rossmann-like demo generator so every component can run immediately. Regenerate final model results using the official files before quoting competition performance.

## Project Overview
Rossmann operates thousands of drug stores. This project predicts daily sales for each store over a six-week horizon to support replenishment, staffing, promotion planning, and short-term financial forecasting.

## Dataset Source
- Kaggle competition: `rossmann-store-sales`
- Training data: 1,017,209 rows and 9 columns
- Public test data: 41,088 rows
- Store metadata: 1,115 stores and 10 columns
- Target: `Sales`
- Competition metric: RMSPE
- Forecast horizon: approximately 6 weeks

## Business Problem
Underestimating sales causes stockouts and missed revenue. Overestimating sales creates excess inventory, labor inefficiency, and working-capital pressure. The model must generalize to future dates, not merely interpolate randomly held-out history.

## Methodology
1. Authenticate to Kaggle and download the competition files.
2. Validate schemas, duplicates, missing values, dates, and store joins.
3. Merge store metadata and impute competition/Promo2 gaps using explicit rules.
4. Engineer calendar, cyclical, promotion, holiday, and competition-duration features.
5. Exclude `Customers`, which is unavailable in the competition test set.
6. Train on historical open-store positive-sales rows using `log1p(Sales)`.
7. Validate and test on consecutive 42-day chronological windows.
8. Compare Ridge Regression and XGBoost using RMSPE, RMSE, MAE, MAPE, and R².
9. Force forecasts to zero when stores are closed.

## Models
### Baseline: Ridge Regression
A transparent regularized model with standardized numeric features and one-hot categorical variables.

### Advanced: XGBoost
A nonlinear boosted-tree model that captures interactions between store type, assortment, promotions, holidays, calendar seasonality, and competition features.

## Evaluation
RMSPE is the primary metric because it matches the Kaggle competition and normalizes errors across stores of different sizes. Supporting absolute metrics help identify operationally large misses.

## Repository Structure
```text
├── app/                       # Multi-page Streamlit application
├── data/raw/                  # Authenticated Kaggle files
├── data/processed/            # Generated clean and feature tables
├── data/sample_data/          # Deterministic sample and preview data
├── docs/                      # Architecture and model card
├── models/                    # Generated Ridge and XGBoost artifacts
├── notebooks/                 # Four end-to-end notebooks
├── presentation/              # 10-slide deck, generator, and notes
├── reports/                   # Metrics, quality reports, and predictions
├── src/                       # Production-style pipeline modules
├── tests/                     # Automated tests
├── visuals/                   # EDA and model charts
├── PROJECT_CHARTER.md
├── EDA_REPORT.md
├── MODEL_COMPARISON.md
├── MODEL_EVALUATION.md
├── BUSINESS_RECOMMENDATIONS.md
├── RESUME_LINKEDIN_INTERVIEW.md
├── run_pipeline.py
└── requirements.txt
```

## Run Locally
```bash
git clone https://github.com/Rachel-Oyeyemi/Rossmann-Store-Sales.git
cd Rossmann-Store-Sales
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Immediate deterministic demo
python run_pipeline.py

# Official Kaggle files
python src/download_data.py --source auto
python run_pipeline.py --train data/raw/train.csv --stores data/raw/store.csv

streamlit run app/app.py
pytest -q
```

## Results
The committed benchmark artifacts are generated from deterministic Rossmann-like demo data and prove that the complete pipeline works. They are not official competition scores.

| Model | Validation RMSPE | Test RMSPE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|
| Ridge Regression | 0.113 | 0.115 | 738 | 522 | 0.937 |
| XGBoost | 0.117 | 0.118 | 789 | 538 | 0.928 |

**Demo recommendation:** Ridge Regression achieved the lower test RMSPE. XGBoost remains the advanced nonlinear challenger and should be retuned and reevaluated on the official data. Run the pipeline on authenticated Kaggle files for final portfolio metrics.

## Business Impact
- Improve inventory and replenishment planning
- Align staffing with expected demand
- Anticipate promotion and holiday demand
- Reduce forecast bias and emergency operational adjustments
- Provide finance with a store-level revenue outlook

## Responsible Use
This project is educational. Production deployment requires current data, prediction intervals, live promotions, inventory and stockout signals, store changes, monitoring, access controls, and human override.

## Future Improvements
- LightGBM/CatBoost and model ensembling
- Quantile forecasting and prediction intervals
- Hierarchical regional reconciliation
- Causal promotion-lift analysis
- Weather and local-event features
- Inventory-aware lost-demand estimation
- Automated drift and forecast-bias monitoring
