import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline import train_pipeline


# ------------------ Configuration ------------------
DATA_PATH = "data/raw/churn.csv"   # relative to project root
TARGET_COL = "Churn"               # must match CSV column name exactly
MODEL_PATH = "artifacts/model.joblib"


# ------------------ Training entry ------------------
def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Optional sanity check
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    print("Splitting features and target...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print("Train-test split (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training pipeline and saving model...")
    os.makedirs("artifacts", exist_ok=True)

    train_pipeline(
        X_train=X_train,
        y_train=y_train,
        save_path=MODEL_PATH
    )

    print("Training completed successfully.")
    print("Model saved at:", MODEL_PATH)


# ------------------ Run ------------------
if __name__ == "__main__":
    main()
