# Business Recommendations

## Executive Summary
Use store-level six-week forecasts as a planning signal rather than an automatic order quantity. Combine predicted demand with safety stock, lead times, inventory availability, promotion plans, and manager overrides.

## Key Findings
1. Sales vary strongly by store, calendar, and promotion status.
2. Store closure is a deterministic business rule and should override model output.
3. Nonlinear interactions justify gradient boosting beyond a linear baseline.
4. `Customers` is predictive but unavailable in the forecast set and must not be used.
5. Error should be monitored by segment, not only as one portfolio average.

## Recommendations
- Create forecast bands for inventory and staffing decisions.
- Review the largest over- and under-forecasts weekly.
- Measure promotion incrementality rather than confusing correlation with causal lift.
- Maintain separate monitoring for high-volume stores and holiday periods.
- Retrain on a regular cadence and after major assortment, store, or promotion-policy changes.

## Risk Assessment
- Historical data is from 2013–2015 and may not transfer to modern behavior.
- Competition and promotion metadata may be incomplete or stale.
- Point forecasts omit uncertainty.
- Store openings, closures, renovations, and stockouts can create structural breaks.
- Forecasts can amplify operational errors if used without human review.

## Future Opportunities
- Quantile forecasts and prediction intervals
- Hierarchical store/region reconciliation
- Causal promotion-uplift modeling
- Inventory-constrained demand estimation
- Automated drift and forecast-bias monitoring
- Holiday and local-event enrichment
