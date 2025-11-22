# app.py - Complete final version (paste this file and run)
import os
import time
import json
import traceback
import streamlit as st
import pandas as pd
import requests
import altair as alt


# Optional local modules (if present in your project)
try:
    from util import load_lottie, stream_data, welcome_message, introduction_message
except Exception:
    load_lottie = None
    stream_data = lambda x: x
    welcome_message = lambda: "Welcome to InsightPilot!"
    introduction_message = lambda: ("InsightPilot helps you clean & analyze data with AI.", "Upload a dataset to get started.")

try:
    from src.util import read_file_from_streamlit
except Exception:
    read_file_from_streamlit = None

try:
    from data_cleaning_agent import clean_data
except Exception:
    clean_data = None

try:
    from data_analysis_agent import analyze_data_with_ai as project_analyze
except Exception:
    project_analyze = None

try:
    from regression_model import regression_model_pipeline
except Exception:
    regression_model_pipeline = None

try:
    from visualization import data_visualization
except Exception:
    data_visualization = None

# If you use python-dotenv for local .env files, optionally load it
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --------------------------
# Legacy model override (guarantee)
# --------------------------
LEGACY_MODEL_MAP = {
    "gemini-1.5-pro": "models/gemini-2.5-pro",
    "gemini-1.5-flash": "models/gemini-2.5-flash",
    "gemini-pro": "models/gemini-2.5-pro",
}

# Ensure safe default in session (do not overwrite if user already set correct model)
if "SELECTED_MODEL" not in st.session_state:
    st.session_state["SELECTED_MODEL"] = "models/gemini-2.5-pro"
else:
    cur = str(st.session_state["SELECTED_MODEL"])
    short = cur.split("/")[-1]
    if short in LEGACY_MODEL_MAP:
        st.session_state["SELECTED_MODEL"] = LEGACY_MODEL_MAP[short]
    elif not cur.startswith("models/"):
        st.session_state["SELECTED_MODEL"] = "models/gemini-2.5-pro"

# --------------------------
# Helpers: API key loader
# --------------------------
def get_gemini_api_key():
    # prefer session (typed in UI)
    key = st.session_state.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    if isinstance(key, str):
        key = key.strip()
        if key == "":
            key = None
    return key

# --------------------------
# Helpers: data coercion (avoid dtype promotion errors)
# --------------------------
def coerce_features_for_regression(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Try to convert datetime-like columns to epoch seconds floats
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                df[col] = (parsed.astype("int64") // 10**9).astype("float64")
        except Exception:
            pass
    # Convert integer dtypes to float to avoid mixed dtype promotion errors
    for col in df.columns:
        try:
            if pd.api.types.is_integer_dtype(df[col].dtype):
                df[col] = df[col].astype("float64")
        except Exception:
            pass
    # Attempt numeric coercion for columns that look numeric
    for col in df.columns:
        if df[col].dtype == "object":
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > len(df) * 0.6:
                df[col] = coerced
    return df

# --------------------------
# Robust Gemini call helpers (try multiple endpoints/payloads)
# --------------------------
def _try_endpoints_with_payloads(endpoints, params, headers, payload_variants, timeout=30):
    last_err = None
    for ep in endpoints:
        for payload in payload_variants:
            try:
                r = requests.post(ep, params=params, headers=headers, json=payload, timeout=timeout)
                if r.status_code == 200:
                    try:
                        return True, r.json()
                    except Exception:
                        return True, r.text
                else:
                    last_err = f"{ep} -> {r.status_code}: {r.text}"
                    # continue to next payload/endpoint
            except Exception as e:
                last_err = f"{ep} request error: {e}"
                continue
    return False, last_err or "No endpoint succeeded."

def _call_generative_with_api_key(model_name: str, prompt: str, api_key: str, max_output_tokens: int = 512):
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent",
        f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent",
        f"https://generativelanguage.googleapis.com/v1/{model_name}:generateText",
        f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateText",
    ]
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    payload_variants = [
        {"input": {"text": prompt}, "maxOutputTokens": max_output_tokens},
        {"prompt": {"text": prompt}, "maxOutputTokens": max_output_tokens},
        {"messages": [{"author": "user", "content": [{"type": "text", "text": prompt}]}], "maxOutputTokens": max_output_tokens},
        {"instances": [{"content": prompt}], "maxOutputTokens": max_output_tokens},
        {"input_text": prompt, "maxOutputTokens": max_output_tokens},
    ]
    ok, resp = _try_endpoints_with_payloads(endpoints, params, headers, payload_variants, timeout=30)
    if ok:
        return True, resp
    return False, f"All attempts failed. Last error: {resp}"

def _call_generative_with_bearer(model_name: str, prompt: str, access_token: str, max_output_tokens: int = 512):
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent",
        f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent",
        f"https://generativelanguage.googleapis.com/v1/{model_name}:generateText",
        f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateText",
    ]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    payload_variants = [
        {"input": {"text": prompt}, "maxOutputTokens": max_output_tokens},
        {"prompt": {"text": prompt}, "maxOutputTokens": max_output_tokens},
        {"messages": [{"author": "user", "content": [{"type": "text", "text": prompt}]}], "maxOutputTokens": max_output_tokens},
        {"instances": [{"content": prompt}], "maxOutputTokens": max_output_tokens},
        {"input_text": prompt, "maxOutputTokens": max_output_tokens},
    ]
    ok, resp = _try_endpoints_with_payloads(endpoints, None, headers, payload_variants, timeout=30)
    if ok:
        return True, resp
    return False, f"All attempts failed. Last error: {resp}"

def _extract_text_from_response(resp_json):
    try:
        if isinstance(resp_json, dict):
            if "candidates" in resp_json:
                return " ".join([c.get("content", "") for c in resp_json.get("candidates", [])])
            if "output" in resp_json:
                out = resp_json.get("output")
                if isinstance(out, list):
                    parts = []
                    for o in out:
                        if isinstance(o, dict):
                            parts.append(o.get("content", "") or o.get("text", "") or "")
                        else:
                            parts.append(str(o))
                    return " ".join(parts)
                return str(out)
            if "results" in resp_json:
                parts = []
                for r in resp_json["results"]:
                    if isinstance(r, dict) and "content" in r:
                        c = r["content"]
                        if isinstance(c, dict):
                            parts.append(c.get("text", "") or c.get("html", "") or "")
                        else:
                            parts.append(str(c))
                if parts:
                    return " ".join(parts)
            if "text" in resp_json:
                return resp_json.get("text", "")
        return json.dumps(resp_json)[:4000]
    except Exception:
        return str(resp_json)[:4000]

# --------------------------
# AI analysis wrapper
# --------------------------
def analyze_data_with_ai_wrapper(df: pd.DataFrame, api_key: str, model_name: str):
    # prefer project analysis if provided
    if project_analyze:
        try:
            out = project_analyze(df, api_key)
            if isinstance(out, tuple):
                return out
            return out, None
        except Exception:
            pass

    # build short prompt
    try:
        cols = df.columns.tolist()
        sample = df.head(5).to_dict(orient="records")
        prompt = f"Analyze this dataset. Columns: {cols}. Sample rows: {sample}. Provide a short statistical summary, top 3 insights, and recommended next steps."
    except Exception:
        prompt = "Analyze dataset and provide summary, 3 insights, and recommendations."

    ok, resp = _call_generative_with_api_key(model_name, prompt, api_key)
    if not ok:
        return None, resp
    try:
        text = _extract_text_from_response(resp)
        results = {
            "ai_response": text,
            "summary": df.describe(include="all"),
            "correlations": df.corr(numeric_only=True),
        }
        return results, None
    except Exception:
        return {"ai_response": resp}, None

# --------------------------
# Streamlit UI (main)
# --------------------------
st.set_page_config(page_title="InsightPilot", page_icon="🚀", layout="wide")

# Header / Intro
with st.container():
    st.subheader("Hello there 👋")
    st.title("InsightPilot")
    try:
        st.write(stream_data(welcome_message()))
    except Exception:
        st.write("Upload a dataset and click Start Analysis.")

with st.container():
    st.divider()
    try:
        intro = introduction_message()
        if isinstance(intro, (list, tuple)):
            for i in intro:
                st.write(i)
        else:
            st.write(intro)
    except Exception:
        st.write("Clean, analyze, and visualize data with AI.")
    st.divider()

# Ensure app stage state exists
if "app_stage" not in st.session_state:
    st.session_state["app_stage"] = "idle"  # idle | processing | done

# Main layout: left = upload/key, right = controls
left_col, right_col = st.columns([6, 4])

# --------------------------
# LEFT: API key + uploader + preview
# --------------------------
with left_col:
    API_KEY = get_gemini_api_key()
    if not API_KEY:
        st.warning("Gemini API key not found in secrets or environment.")
        entered_key = st.text_input("Enter your Gemini API key (kept only for this session)", type="password", key="widget_api_key_input")
        if entered_key:
            st.session_state["GEMINI_API_KEY"] = entered_key.strip()
            API_KEY = st.session_state["GEMINI_API_KEY"]
            st.success("API key saved for this session.")
    else:
        st.write("🔒 Gemini API key loaded for this session.")

    st.write("Upload a data file (CSV / JSON / XLSX). Your data is not stored.")
    uploaded_file = st.file_uploader("Upload data file", type=["csv", "json", "xls", "xlsx"], key="widget_main_uploader")

    if uploaded_file is not None:
        try:
            if read_file_from_streamlit:
                df = read_file_from_streamlit(uploaded_file)
            else:
                import io
                name = uploaded_file.name.lower()
                buf = uploaded_file.getbuffer()
                if name.endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(buf))
                elif name.endswith(".json"):
                    df = pd.read_json(io.BytesIO(buf))
                else:
                    df = pd.read_excel(io.BytesIO(buf))
            if df is None or df.shape[0] == 0:
                st.error("Uploaded file is empty or could not be read.")
                st.session_state["is_file_empty"] = True
            else:
                st.session_state["DF_uploaded"] = df
                st.session_state["is_file_empty"] = False
                st.success("File uploaded successfully!")
                st.write("### Data preview")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            st.session_state["is_file_empty"] = True

# --------------------------
# RIGHT: model/mode/select/start/test
# --------------------------
with right_col:
    MODEL_OPTIONS = {
        "Gemini 2.5 Pro (recommended)": "models/gemini-2.5-pro",
        "Gemini 2.5 Flash": "models/gemini-2.5-flash",
        "Gemini 2.0 Flash": "models/gemini-2.0-flash",
        "Gemini 2.0 Flash 001": "models/gemini-2.0-flash-001",
    }

    friendly_choice = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()), key="widget_model_select")
    selected_model_ui = MODEL_OPTIONS[friendly_choice]
    # ensure session-safe override for legacy short names
    short = str(selected_model_ui).split("/")[-1]
    if short in LEGACY_MODEL_MAP:
        st.session_state["SELECTED_MODEL"] = LEGACY_MODEL_MAP[short]
    else:
        st.session_state["SELECTED_MODEL"] = selected_model_ui

    MODE = st.selectbox("Select data analysis mode", ["Regression Model", "Data Visualization"], key="widget_mode_select")
    st.write(f"Model: {friendly_choice}")
    st.write(f"Mode: {MODE}")

    # Start button: set stage to processing, but do NOT call st.rerun()
    start_disabled = not ("DF_uploaded" in st.session_state and not st.session_state.get("is_file_empty", True) and API_KEY)
    if st.button("🚀 Start Analysis", disabled=start_disabled, key="widget_start_button"):
        # clear previous results for a fresh run
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("cleaned_df", None)
        st.session_state["app_stage"] = "processing"

    st.markdown("---")
    st.markdown("### 🔍 Credential test")
    test_key = st.text_input("Quick test: paste API key here (optional)", type="password", key="widget_test_key")
    test_sa = st.text_input("Service account JSON path on server (optional)", key="widget_test_sa")
    if st.button("Test Credentials", key="widget_test_credentials"):
        if test_sa:
            try:
                from google.oauth2 import service_account
                import google.auth.transport.requests
                scopes = ["https://www.googleapis.com/auth/cloud-platform"]
                creds = service_account.Credentials.from_service_account_file(test_sa, scopes=scopes)
                auth_req = google.auth.transport.requests.Request()
                creds.refresh(auth_req)
                token = creds.token
                st.success("Got access token (truncated): " + token[:80] + "...")
                ok, resp = _call_generative_with_bearer(st.session_state.get("SELECTED_MODEL"), "Say hello", token)
                if ok:
                    st.write(json.dumps(resp)[:2000])
                else:
                    st.error(resp)
            except Exception as e:
                st.error(f"Service-account test failed: {e}")
        elif test_key:
            ok, resp = _call_generative_with_api_key(st.session_state.get("SELECTED_MODEL"), "Say hello", test_key)
            if ok:
                st.write(json.dumps(resp)[:2000])
            else:
                st.error(resp)
        else:
            st.info("Provide an API key or service-account JSON path to test.")

# --------------------------
# PROCESSING: run once per Start click; caches results in session
# --------------------------
if st.session_state.get("app_stage") == "processing":
    API_KEY = get_gemini_api_key()
    MODEL = st.session_state.get("SELECTED_MODEL", "models/gemini-2.5-pro")

    # UI placeholders (no keys)
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text("🧹 Cleaning data...")
        time.sleep(0.3)
        if "DF_uploaded" not in st.session_state:
            st.error("No dataset found. Please upload a file first.")
            st.session_state["app_stage"] = "idle"
        else:
            df = st.session_state["DF_uploaded"]

            # clean once and cache
            if "cleaned_df" not in st.session_state:
                try:
                    cleaned = clean_data(df) if clean_data else df.copy()
                except Exception:
                    cleaned = df.copy()
                try:
                    cleaned = coerce_features_for_regression(cleaned)
                except Exception:
                    pass
                st.session_state["cleaned_df"] = cleaned
            else:
                cleaned = st.session_state["cleaned_df"]

            progress.progress(25)

            # AI analysis: run once and store results
            status.text("📊 Running AI-powered data analysis...")
            time.sleep(0.3)
            if "analysis_results" not in st.session_state:
                results, err = analyze_data_with_ai_wrapper(cleaned, api_key=API_KEY, model_name=MODEL)
                if results is None:
                    st.error(f"AI analysis failed: {err}")
                    st.session_state["app_stage"] = "idle"
                else:
                    st.session_state["analysis_results"] = results
                    st.session_state["app_stage"] = "done"
            else:
                # already have results (should not usually happen here)
                pass

            progress.progress(70)
            status.text("✅ Analysis done. See results below.")
            time.sleep(0.2)
            progress.progress(100)
    except Exception as e:
        st.error(f"Unexpected error during processing: {e}")
        st.write(traceback.format_exc())
        st.session_state["app_stage"] = "idle"
        progress.progress(0)

# --------------------------
# RESULTS & VISUALIZATION (switching viz won't restart pipeline)
# --------------------------
if st.session_state.get("app_stage") == "done" and "analysis_results" in st.session_state:
    results = st.session_state["analysis_results"]
    cleaned = st.session_state.get("cleaned_df", None)

    # Summary & AI output
    with st.expander("🔍 View Analysis Summary", expanded=True):
        st.subheader("📄 Statistical Summary")
        try:
            summ = results.get("summary")
            if isinstance(summ, (pd.DataFrame, pd.Series)):
                st.dataframe(summ)
            else:
                try:
                    st.dataframe(pd.DataFrame(summ))
                except Exception:
                    st.write(summ)
        except Exception:
            st.write("No summary available.")

        st.subheader("📈 Correlation Matrix")
        try:
            corr = results.get("correlations")
            if isinstance(corr, pd.DataFrame):
                st.dataframe(corr)
            else:
                try:
                    st.dataframe(pd.DataFrame(corr))
                except Exception:
                    st.write("No correlation matrix available.")
        except Exception:
            st.write("No correlation matrix available.")

        st.subheader("🧠 AI Insights & Recommendations")
        st.write(results.get("ai_response", "No AI response available."))

    st.divider()

    # Visualization controls (stateful, safe)
    st.subheader("Data Visualization")
    if cleaned is None:
        st.info("No cleaned data available for visualization. Run analysis first.")
    else:
        # ensure viz_type is preserved across interactions
        if "viz_type" not in st.session_state:
            st.session_state["viz_type"] = "Histogram"

        viz_choice = st.selectbox("Choose visualization", ["Histogram", "Scatter", "Boxplot", "Table"], index=["Histogram", "Scatter", "Boxplot", "Table"].index(st.session_state["viz_type"]), key="widget_viz_select")
        st.session_state["viz_type"] = viz_choice

        numeric_cols = cleaned.select_dtypes(include="number").columns.tolist()
        all_cols = cleaned.columns.tolist()

        if viz_choice == "Histogram":
            if numeric_cols:
                hist_col = st.selectbox("Column for histogram", numeric_cols, key="widget_hist_col")
                st.write(f"Histogram — {hist_col}")
                st.bar_chart(cleaned[hist_col].dropna())
            else:
                st.info("No numeric columns available for histogram.")
        elif viz_choice == "Scatter":
            if len(numeric_cols) >= 2:
                xcol = st.selectbox("X-axis", numeric_cols, index=0, key="widget_scatter_x")
                ycol = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="widget_scatter_y")
                st.write(f"Scatter — X: {xcol}, Y: {ycol}")

                chart_df = cleaned[[xcol, ycol]].dropna()
                if chart_df.empty:
                    st.info("No rows available after dropping NaNs.")
                else:
                    # Try to parse x as datetime for better plotting if it looks like a date
                    try:
                        if not pd.api.types.is_datetime64_any_dtype(chart_df[xcol]):
                            parsed = pd.to_datetime(chart_df[xcol], errors="coerce")
                            if parsed.notna().sum() > 0:
                                chart_df[xcol] = parsed
                    except Exception:
                        pass

                    x_type = "quantitative" if pd.api.types.is_numeric_dtype(chart_df[xcol]) else "temporal"
                    alt_chart = alt.Chart(chart_df).mark_point().encode(
                        x=alt.X(xcol, type=x_type),
                        y=alt.Y(ycol, type="quantitative"),
                        tooltip=list(chart_df.columns)
                    ).interactive().properties(height=450)
                    st.altair_chart(alt_chart, use_container_width=True)
            else:
                st.info("Need at least two numeric columns for scatter plot.")


        
        # elif viz_choice == "Scatter":
        #     if len(numeric_cols) >= 2:
        #         xcol = st.selectbox("X-axis", numeric_cols, index=0, key="widget_scatter_x")
        #         ycol = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols)>1 else 0, key="widget_scatter_y")
        #         st.write(f"Scatter — X: {xcol}, Y: {ycol}")
        #         chart = cleaned[[xcol, ycol]].dropna()
        #         st.altair_chart(st.vega_lite_chart if False else st.line_chart(chart) )  # simple fallback
        #         # For better scatter use: st.altair_chart(...) or matplotlib
        #     else:
        #         st.info("Need at least two numeric columns for scatter plot.")
        elif viz_choice == "Boxplot":
            if numeric_cols:
                box_col = st.selectbox("Column for boxplot", numeric_cols, key="widget_box_col")
                st.write(f"Boxplot — {box_col}")
                st.write(cleaned[box_col].dropna().describe())
                # Could use matplotlib or altair for a real boxplot
            else:
                st.info("No numeric columns available for boxplot.")
        else:  # Table
            st.write("Data table (first 200 rows):")
            st.dataframe(cleaned.head(200))

# Footer tips
st.markdown("---")
st.caption("Tip: change visualization options freely — the app will not redirect you to the start page. If you change API key/model, re-run Start Analysis to refresh AI results.")
