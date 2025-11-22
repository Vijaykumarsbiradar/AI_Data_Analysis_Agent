import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from google.generativeai import GenerativeModel, configure
import os
from dotenv import load_dotenv
import streamlit as st

# --- Load API key from .env (fallback if not passed from app) ---
load_dotenv()

def analyze_data_with_ai(df, api_key=None):
    """Analyze dataset, explain insights, detect anomalies, and suggest visuals using Gemini API."""
    if df is None or df.empty:
        st.warning("⚠️ No data provided for analysis.")
        return None, None

    # === Initialize Gemini client ===
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        st.error("❌ Gemini API key not found! Please provide a valid key.")
        return None, None

    configure(api_key=key)
    model = GenerativeModel("models/gemini-2.5-pro")
    st.write("📊 Starting AI-powered data analysis (Gemini)...")

    # === Step 1: Summary Statistics ===
    summary = df.describe(include='all')
    num_data = df.select_dtypes(include='number')

    # === Step 2: Clustering (optional) ===
    if not num_data.empty:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(num_data)
        kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled)
        cluster_summary = df['Cluster'].value_counts().to_dict()
    else:
        cluster_summary = {}

    # === ✅ Step 3: Anomaly Detection ===
    anomaly_summary = {}
    if not num_data.empty:
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso.fit_predict(num_data)
        df['Anomaly'] = anomalies  # -1 = anomaly, 1 = normal
        anomaly_count = (df['Anomaly'] == -1).sum()
        anomaly_summary = {
            "Total Anomalies": int(anomaly_count),
            "Percentage": round((anomaly_count / len(df)) * 100, 2)
        }
    else:
        df['Anomaly'] = None

    # === Step 4: Correlation Matrix ===
    correlations = num_data.corr() if not num_data.empty else pd.DataFrame()

    # === Step 5: Context summary for Gemini ===
    analysis_text = f"""
    Dataset shape: {df.shape}
    Columns: {list(df.columns)}
    Missing values per column:
    {df.isnull().sum().to_dict()}

    Statistical Summary:
    {summary.to_string()}

    Cluster Summary:
    {cluster_summary}

    Anomaly Summary:
    {anomaly_summary}

    Correlation Matrix:
    {correlations.to_string()}
    """

    # === Step 6: Ask Gemini for insights ===
    prompt = f"""
    You are a professional data analyst.
    Analyze the following dataset summary, correlation matrix, clustering, and anomaly detection report.
    - Explain what the data indicates.
    - Mention any outliers or unusual points if anomaly count is high.
    - Suggest 3 improvements to data quality or structure.
    - Recommend 3 best visualization types (e.g. heatmap, bar, scatter, histogram, boxplot, pie, line).

    Dataset Info:
    {analysis_text}
    """

    try:
        response = model.generate_content(prompt)
        ai_response = response.text.strip()
    except Exception as e:
        ai_response = f"⚠️ Error getting AI insights: {e}"

    # === Step 7: Extract suggested chart types ===
    suggested_charts = []
    for chart_type in ["heatmap", "bar", "scatter", "histogram", "box", "pie", "line"]:
        if chart_type in ai_response.lower():
            suggested_charts.append(chart_type)

    # === Step 8: Return analysis summary ===
    results = {
        "summary": summary,
        "ai_response": ai_response,
        "suggested_charts": suggested_charts,
        "cluster_summary": cluster_summary,
        "anomaly_summary": anomaly_summary,
        "correlations": correlations,
        "processed_df": df
    }

    return results, summary
