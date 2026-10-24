import argparse
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file"""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df = df.drop_duplicates()

    if "customerID" in df.columns:          # ✅ fixed: lowercase c
        df = df.drop(columns=["customerID"])

    if df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    for col in df.select_dtypes(include=["object", "string"]).columns:   # ✅ fixed spelling
        df[col] = df[col].str.strip()

    df = df.reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean the Telco Customer Churn dataset")
    parser.add_argument("--input", default="churn.csv", help="Path to raw CSV file")
    parser.add_argument("--output", default="cleaned_churn.csv", help="Path to save cleaned CSV")
    args = parser.parse_args()

    print(f"Loading raw data from {args.input} ...")
    df_raw = load_data(args.input)
    print(f"Raw shape: {df_raw.shape}")

    df_clean = clean_data(df_raw)             
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")

    df_clean.to_csv(args.output, index=False)   
    print(f"Cleaning done. File saved as {args.output}")


if __name__ == "__main__":
    main()