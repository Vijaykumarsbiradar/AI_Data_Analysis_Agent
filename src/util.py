import pandas as pd
import io

def read_file_from_streamlit(uploaded_file):
    """
    Reads an uploaded file (CSV, Excel, or JSON) from Streamlit's file uploader
    and returns a pandas DataFrame.

    Parameters:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        pd.DataFrame: The loaded data as a DataFrame
    """
    try:
        file_name = uploaded_file.name.lower()

        # === CSV FILE ===
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        # === EXCEL FILE (XLS or XLSX) ===
        elif file_name.endswith(".xls") or file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        # === JSON FILE ===
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)

        # === UNKNOWN FORMAT ===
        else:
            raise ValueError("Unsupported file type. Please upload CSV, Excel, or JSON files only.")

        # Strip extra whitespace from column names
        df.columns = df.columns.str.strip()

        # Remove entirely empty rows and columns
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        return df

    except Exception as e:
        raise ValueError(f"❌ Error reading file: {e}")
