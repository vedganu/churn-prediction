"""
Preprocessing: clean the churn data, split it into training and test parts,
and convert the text and numbers into a form the models can use.

Saves:
- training and test files (already converted to numbers)
- the preprocessor, so new customers can be prepared the same way
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
ID_COLUMNS = ["customerID"]
RANDOM_STATE = 42


def load_and_clean(input_path):
    """Load the data and make the target and numeric columns usable."""
    df = pd.read_csv(input_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    # Drop ID columns. They are labels, not useful signals.
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # TotalCharges is sometimes stored as text, with blanks for new customers.
    # Convert it to numbers. Anything unreadable becomes a gap (NaN).
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Turn Yes/No into 1/0. Any other value becomes NaN, which we check below.
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    if df[TARGET_COL].isna().any():
        raise ValueError("Churn column must contain only 'Yes' or 'No'.")

    return df


def get_feature_lists(X):
    """Sort the input columns into numbers and text."""
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """Rules for cleaning: fill gaps, scale numbers, turn text into 0/1 columns."""
    numeric_pipe = Pipeline([
        ("fill_gaps", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("fill_gaps", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def run_preprocessing(input_path, output_dir, model_dir):
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(input_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 80% for training, 20% hidden until the final test.
    # stratify keeps the same churn share in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Churn rate, train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")

    # Learn the cleaning rules from the training data only.
    # The test data is only transformed, never used to set any rule.
    numeric_cols, categorical_cols = get_feature_lists(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_t = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_t = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)
    print(f"Number of input columns after conversion: {X_train_t.shape[1]}")

    # Save the converted data and the answers.
    X_train_t.to_csv(output_dir / "X_train_scaled.csv", index=False)
    X_test_t.to_csv(output_dir / "X_test_scaled.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    # Save the preprocessor so future customers are cleaned the same way.
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")
    print("Preprocessing done.")


def main():
    parser = argparse.ArgumentParser(description="Prepare the churn data for modelling.")
    parser.add_argument("--input", default="data/processed/cleaned_churn.csv",
                        help="Path to the cleaned churn CSV.")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Folder for the train/test CSV files.")
    parser.add_argument("--model-dir", default="models",
                        help="Folder for the saved preprocessor.")
    args = parser.parse_args()

    run_preprocessing(args.input, args.output_dir, args.model_dir)


if __name__ == "__main__":
    main()