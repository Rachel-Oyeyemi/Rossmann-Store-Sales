# Resume, LinkedIn, and Interview Materials

## Resume Bullet Points
- Built an end-to-end retail sales forecasting pipeline for 1M+ historical store-day records, combining schema validation, metadata imputation, leakage-safe feature engineering, chronological validation, and reproducible model artifacts.
- Compared a regularized Ridge baseline with XGBoost using Kaggle RMSPE, RMSE, MAE, MAPE, and R², while excluding inference-time leakage and enforcing zero forecasts for closed stores.
- Developed a multi-page Streamlit forecasting application, automated tests, CI workflow, executive presentation, and business recommendations for inventory, staffing, promotion, and forecast monitoring.

## LinkedIn Project Description
I built a production-style Rossmann Store Sales forecasting portfolio project focused on predicting daily store sales six weeks ahead. The solution includes authenticated Kaggle ingestion, data-quality checks, store-metadata merging, leakage-safe calendar and promotion features, chronological validation, Ridge and XGBoost models, RMSPE-based evaluation, a Streamlit prediction app, automated testing, visual reporting, and executive recommendations. A key design decision was excluding `Customers`, which is highly predictive but unavailable in the competition test set. The project demonstrates not only model development, but also reproducibility, responsible validation, and translation of forecasts into inventory and staffing decisions.

## Interview Talking Points
- Why time-based validation is mandatory for forecasting
- How `Customers` creates leakage
- Why RMSPE is useful and where it can be unstable
- How closed-store logic should override the model
- Why log-transforming sales helps
- How promotion, holidays, store type, assortment, and competition interact
- How to monitor forecast bias and error concentration

## Common Questions and Sample Answers
### Why not use a random split?
A random split exposes the model to future seasonal patterns during training and produces an unrealistically optimistic estimate. I used consecutive six-week windows to match the real forecasting horizon.

### Why exclude Customers?
Customer count is observed after or during the sales day and is absent from the Kaggle test set. Using it would make training performance look strong but the model impossible to use at forecast time.

### Why compare Ridge with XGBoost?
Ridge establishes a transparent additive baseline. XGBoost captures nonlinear interactions. The comparison demonstrates whether complexity produces a measurable out-of-time improvement.

### Why RMSPE?
It is the competition metric and normalizes errors across stores with different sales levels. I also report absolute metrics because percentage metrics can behave poorly near zero.

### What would you add in production?
Current promotions, inventory and stockout signals, local events, weather, store changes, uncertainty intervals, hierarchical reconciliation, drift monitoring, and a governed retraining process.
