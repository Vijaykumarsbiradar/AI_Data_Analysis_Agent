# 🧠 AI Data Analysis Agent

An end-to-end **AI-powered data analysis assistant** that helps you go from raw CSV files to clean data, insights, machine-learning models, and exportable reports.

This project combines multiple specialized Python agents:

- 🔧 **Data Cleaning Agent** – handles missing values, duplicates, type conversions, and basic preprocessing
- 📊 **Data Analysis Agent** – performs EDA (summary stats, correlations, distributions)
- 📈 **Regression Model Agent** – trains and evaluates regression models on numeric targets
- 📝 **Report Export Agent** – generates ready-to-share analysis reports

All of this is orchestrated from a single entry point (`app.py`).

---

## 🚀 Features

- Upload and process tabular data (e.g., CSV)
- Automatic data cleaning (null handling, encoding, scaling, etc.)
- Descriptive statistics and EDA
- Regression model training & evaluation (e.g., RMSE, MAE, R²)
- Simple visualizations (histograms, correlations, line plots, etc.)
- Exportable reports (e.g., text/HTML/PDF – depending on your implementation)
- Modular design – each agent is a separate Python module

---

## 🏗 Project Structure

```bash
AI_Data_Analysis_Agent/
├── app.py                   # Main entry point / orchestrator
├── data_analysis_agent.py   # Logic for EDA & insights
├── data_cleaning_agent.py   # Logic for cleaning & preprocessing data
├── regression_model.py      # Model training & evaluation utilities
├── report_export_agent.py   # Report generation & export logic
├── visualization.py         # Plotting & visualization helpers
├── util.py                  # Shared helper functions
├── list_models.py           # Utility to list / manage available models
├── requirements.txt         # Python dependencies
└── src/                     # (Optional) extra source modules, if used
