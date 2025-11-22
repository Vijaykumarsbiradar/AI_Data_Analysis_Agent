import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests
import os

def data_visualization(df, ai_response=None, api_key=None):
    """Generate AI-enhanced data visualizations with explanations (Gemini version)."""
    st.subheader("📊 Data Visualization Dashboard")

    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(exclude='number').columns

    chart_type = st.selectbox(
        "📌 Choose a chart type:",
        ["Histogram", "Boxplot", "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Correlation Heatmap"]
    )

    # --- Chart Generation ---
    if chart_type == "Histogram":
        col = st.selectbox("Select numeric column:", num_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    elif chart_type == "Boxplot":
        col = st.selectbox("Select numeric column:", num_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax, color='lightgreen')
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    elif chart_type == "Bar Chart":
        col = st.selectbox("Select categorical column:", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax, color='orange')
        ax.set_title(f"Bar Chart of {col}")
        st.pyplot(fig)

    elif chart_type == "Line Chart":
        x = st.selectbox("Select X-axis:", num_cols)
        y = st.selectbox("Select Y-axis:", num_cols)
        fig, ax = plt.subplots()
        ax.plot(df[x], df[y], color='purple')
        ax.set_title(f"Trend Line: {y} over {x}")
        st.pyplot(fig)

    elif chart_type == "Scatter Plot":
        x = st.selectbox("Select X-axis:", num_cols)
        y = st.selectbox("Select Y-axis:", num_cols)
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y], alpha=0.6, color='coral')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Scatter Plot: {x} vs {y}")
        st.pyplot(fig)

    elif chart_type == "Pie Chart":
        col = st.selectbox("Select categorical column:", cat_cols)
        counts = df[col].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
        ax.set_title(f"Pie Chart of {col}")
        st.pyplot(fig)

    elif chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    # --- 🧠 Gemini Trend Explanation (optional) ---
    if ai_response and api_key:
        try:
            prompt = f"""
            You are a professional data analyst.
            Explain the major trend or insight visible in this {chart_type} based on the dataset description below:
            {ai_response}
            """

            # use the supported v1 endpoint and model id
            url = f"https://generativelanguage.googleapis.com/v1/models/models/gemini-2.5-pro:generateText?key={api_key}"
            headers = {"Content-Type": "application/json"}
            # v1 generateText commonly accepts a 'prompt' shape
            payload = {"prompt": {"text": prompt}, "maxOutputTokens": 256, "temperature": 0.2}

            response = requests.post(url, headers=headers, json=payload, timeout=20)

            if response.status_code == 200:
                result = response.json()
                explanation = result["candidates"][0]["content"]["parts"][0]["text"]
                st.markdown("### 🧠 AI-Generated Trend Explanation (Gemini)")
                st.info(explanation)
            else:
                st.warning(f"⚠️ Gemini API error: {response.status_code} - {response.text}")

        except Exception as e:
            st.warning(f"⚠️ Could not generate AI explanation: {e}")
