# Walmart Sales Forecasting

A machine learning project that forecasts weekly Walmart sales using historical sales data, store-level trends, seasonality, and holiday effects. The dashboard supports forecasts for individual stores or all stores combined, up to 52 weeks into the future.

## Live Demo

**[Open the Walmart Sales Forecasting Dashboard](https://walmart-sales-nine.vercel.app/)**

Try the deployed dashboard to explore weekly sales forecasts, store comparisons, and holiday impact insights.


*Dashboard preview showing weekly sales forecasts, historical trends, top-performing stores, and holiday impact insights.*

## Project Overview

Retail sales vary over time due to seasonal patterns, store-level differences, and major shopping holidays. This project uses historical Walmart sales records to estimate future weekly sales and present the results in an interactive dashboard.

The system was trained using historical data from **45 Walmart stores** and is designed to support sales planning and trend analysis.

## Key Features

- **Forecast up to 52 weeks ahead:** Generate weekly sales predictions for future planning.
- **Store-level forecasting:** View forecasts for a selected store or all stores combined.
- **Historical trend comparison:** Compare forecasted sales with historical weekly sales.
- **Holiday impact analysis:** Explore sales patterns around Thanksgiving, Christmas, the Super Bowl, and Labor Day.
- **Store comparison:** Review stores with the highest predicted sales.
- **Interactive dashboard:** View key forecast metrics and generate a report from the interface.

## Model Performance

The model was evaluated on a **chronological holdout set** consisting of the final 20% of recorded weeks. This time-based split is intended to evaluate performance on later observations that were not used for training.

| Metric | Reported result | What it means |
|---|---:|---|
| Prediction accuracy* | 95.8% | Reported as 100% minus MAPE |
| R² score | 0.984 | The model explains approximately 98.4% of the variation in the evaluated target values |
| Mean Absolute Error (MAE) | $40.7K | Average absolute prediction error, in the target's units |
| Mean Absolute Percentage Error (MAPE) | 4.16% | Average absolute percentage error |

## Technology Stack

- **Core & Runtime:** Python 3
- **Machine Learning:** 
  - `scikit-learn` – Linear Regression, `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `permutation_importance`
  - `joblib` – Model serialization and pipeline persistence
- **Data Engineering & Analysis:** 
  - `pandas` – Time-series preprocessing, rolling averages, lag feature generation
  - `numpy` – Vectorized numerical operations and statistical evaluations
- **Backend & API:** 
  - `Flask` – RESTful endpoints for metrics, forecasts, and CSV report export
  - `Waitress` – Multi-threaded production WSGI server
- **Frontend & Dashboard:** 
  - HTML5 & CSS3 (Fully responsive, dark-mode design with CSS Grid / Flexbox)
  - Vanilla JavaScript (ES6+)
  - `Chart.js` – Interactive historical vs. forecast line charts and holiday lift bar charts
  - `Lucide Icons` – UI vector icons


## How It Works

1. **Load historical data** — read the recorded weekly sales and relevant store or holiday information.
2. **Prepare the data** — clean and transform the data and prepare the model features.
3. **Use a chronological split** — reserve the latest 20% of recorded weeks for testing.
4. **Train and evaluate the model** — fit the forecasting model on earlier data and calculate metrics on the holdout period.
5. **Generate forecasts** — predict future weekly sales for a selected store or across stores.
6. **Visualize the results** — display forecasts, historical trends, store comparisons, and holiday insights in the dashboard.

## Business Use Cases

- Support weekly and seasonal sales planning.
- Help compare expected sales across stores.
- Identify periods associated with higher or lower sales.
- Provide a starting point for inventory and staffing discussions.

Forecasts are estimates and should be considered alongside business context and actual sales results.

