# Project Charter — Rossmann Store Sales

## Business Problem
Rossmann planners need store-level daily sales forecasts six weeks ahead to coordinate inventory, staffing, promotion execution, cash flow, and regional performance management. Under-forecasting creates stockouts and lost sales; over-forecasting creates excess inventory and avoidable labor or working-capital costs.

## Objectives
1. Build a reproducible pipeline from competition files to validated forecasts.
2. Compare a transparent regularized linear baseline with nonlinear gradient boosting.
3. Prevent leakage through chronological splits and exclusion of `Customers`.
4. Translate error metrics into store-planning decisions and monitoring requirements.

## Stakeholders
- Store operations and regional managers
- Inventory and replenishment teams
- Promotion and merchandising teams
- Finance and workforce-planning leaders
- Data science, engineering, and model-risk teams

## Success Metrics
- Primary: RMSPE on open stores with positive actual sales
- Supporting: RMSE, MAE, MAPE, R²
- Operational: forecast bias, stockout rate, inventory turns, staffing variance, and promotion forecast accuracy

## Expected Business Impact
A reliable forecast can improve replenishment, reduce emergency transfers, align staffing with demand, quantify promotion lift, and give finance a more accurate short-term revenue view.

## Technical Architecture
Kaggle authentication → raw CSV validation → store-metadata merge → calendar/promotion/competition features → chronological train/validation/test windows → Ridge and XGBoost → RMSPE evaluation → saved artifacts → Streamlit app and executive reports.

## End-to-End Workflow
1. Accept competition rules and download `train.csv`, `test.csv`, and `store.csv`.
2. Validate schemas, types, duplicates, missing metadata, and date coverage.
3. Impute competition and Promo2 metadata using explicit business rules.
4. Engineer only features available at forecast time.
5. Train on historical open-store positive-sales rows.
6. Validate and test on consecutive six-week time windows.
7. Force predictions to zero for closed stores.
8. Publish metrics, model artifacts, visuals, app, documentation, and monitoring recommendations.

## Scope Boundaries
This is a portfolio and competition-style forecast. Production use requires live calendars, store changes, promotion plans, inventory constraints, uncertainty intervals, and governed retraining.
