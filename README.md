Walmart Sales Forecasting
A machine learning project that forecasts weekly Walmart sales using historical sales data, store-level trends, seasonality, and holiday effects. The dashboard supports forecasts for individual stores or all stores combined, up to 52 weeks into the future.
Live Demo
Open the Walmart Sales Forecasting Dashboard

Try the deployed dashboard to explore weekly sales forecasts, store comparisons, and holiday impact insights.
Dashboard Preview

Dashboard preview showing weekly sales forecasts, historical trends, top-performing stores, and holiday impact insights.
Project Overview
Retail sales vary over time due to seasonal patterns, store-level differences, and major shopping holidays. This project uses historical Walmart sales records to estimate future weekly sales and present the results in an interactive dashboard.

The system was trained using historical data from 45 Walmart stores and is designed to support sales planning and trend analysis.
Key Features
- Forecast up to 52 weeks ahead: Generate weekly sales predictions for future planning.
- Store-level forecasting: View forecasts for a selected store or all stores combined.
- Historical trend comparison: Compare forecasted sales with historical weekly sales.
- Holiday impact analysis: Explore sales patterns around Thanksgiving, Christmas, the Super Bowl, and Labor Day.
- Store comparison: Review stores with the highest predicted sales.
- Interactive dashboard: View key forecast metrics and generate a report from the interface.
Model Performance
The model was evaluated on a chronological holdout set consisting of the final 20% of recorded weeks. This time-based split is intended to evaluate performance on later observations that were not used for training.
Metric	Reported result	What it means
Prediction accuracy*	95.8%	Reported as 100% minus MAPE
R² score	0.984	The model explains approximately 98.4% of the variation in the evaluated target values
Mean Absolute Error (MAE)	$40.7K	Average absolute prediction error, in the target's units
Mean Absolute Percentage Error (MAPE)	4.16%	Average absolute percentage error


*The 95.8% figure is a project-defined metric calculated as 100% − MAPE; it is not a universal forecasting accuracy measure. Confirm that the displayed metrics were calculated on the same test set and target aggregation before comparing them.

Important: MAE depends on the scale of the prediction target. If predictions are aggregated across stores, explain that aggregation clearly so readers can interpret the $40.7K error correctly.
Technology Stack
- Language: Python
- Machine learning: Linear Regression (as displayed in the dashboard)
- Data analysis: Add the libraries actually used in your code, such as pandas and NumPy
- Visualization / dashboard: Add the framework and charting libraries actually used in your application
Update this section to match the project's actual imports and implementation. The dashboard screenshot displays Linear Regression with an R² value of 0.984.

How It Works
1. Load historical data — read the recorded weekly sales and relevant store or holiday information.
2. Prepare the data — clean and transform the data and prepare the model features.
3. Use a chronological split — reserve the latest 20% of recorded weeks for testing.
4. Train and evaluate the model — fit the forecasting model on earlier data and calculate metrics on the holdout period.
5. Generate forecasts — predict future weekly sales for a selected store or across stores.
6. Visualize the results — display forecasts, historical trends, store comparisons, and holiday insights in the dashboard.
Business Use Cases
- Support weekly and seasonal sales planning.
- Help compare expected sales across stores.
- Identify periods associated with higher or lower sales.
- Provide a starting point for inventory and staffing discussions.
Forecasts are estimates and should be considered alongside business context and actual sales results.
