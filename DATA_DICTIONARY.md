# Data Dictionary

## Training Table
| Column | Meaning | Modeling treatment |
|---|---|---|
| Store | Store identifier | Categorical |
| DayOfWeek | 1–7 day index | Categorical plus cyclical encoding |
| Date | Calendar date | Parsed and expanded |
| Sales | Daily sales | Regression target |
| Customers | Daily customer count | Excluded: unavailable at inference |
| Open | Store open flag | Feature and zero-forecast business rule |
| Promo | Standard promotion flag | Numeric flag |
| StateHoliday | State-holiday category | Categorical |
| SchoolHoliday | School-holiday flag | Numeric flag |

## Store Metadata
| Column | Meaning |
|---|---|
| StoreType | Store format category |
| Assortment | Product assortment level |
| CompetitionDistance | Distance to nearest competitor |
| CompetitionOpenSinceMonth/Year | Approximate competition start |
| Promo2 | Extended promotion participation |
| Promo2SinceWeek/Year | Extended promotion start |
| PromoInterval | Months in which Promo2 repeats |
