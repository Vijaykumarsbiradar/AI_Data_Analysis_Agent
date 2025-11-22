#regression_model
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def regression_model_pipeline(df, api_key=None, gpt_model=None):
    """Perform regression analysis."""
    st.subheader("📈 Regression Model Results")

    target_col = st.selectbox("Select Target Column (Y):", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert X to numeric (one-hot encoding for categorical variables)
    X = pd.get_dummies(X, drop_first=True)

    # Convert y to numeric if needed
    if y.dtype not in ['float64', 'int64']:
        try:
            y = pd.to_numeric(y)
        except Exception:
            try:
                y = pd.factorize(y)[0]
            except Exception:
                st.error("❌ Target column cannot be converted to numeric. Please select a valid numeric target.")
                return

    # Check for empty columns
    if X.shape[1] == 0:
        st.error("❌ No valid features available after encoding. Please check your data.")
        return

    # Model training
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        st.write(f"✅ **R² Score:** {r2:.2f}")
        st.write(f"✅ **MSE:** {mse:.2f}")

    except Exception as e:
        st.error(f"❌ Regression error: {e}")

