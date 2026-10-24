"""
Churn prediction: train a model that flags customers likely to leave.

Steps:
1. Load the cleaned data.
2. Split it into a training part and a hidden test part.
3. Try two models and keep the better one.
4. Pick a cut-off for "will churn".
5. Test once on the hidden part and save the results.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Settings (change here only) ----
# Relative path: run this script from the Churn_Prediction_Project folder.
DATA_FILE = Path("data/processed/cleaned_churn.csv")
MODEL_FILE = Path("models/churn_model.pkl")
REPORT_FILE = Path("reports/results.json")
TARGET = "Churn"
ID_COLUMNS = ["customerID"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
RANDOM_STATE = 42


def load_data():
    """Load the data and split it into training and test parts."""
    df = pd.read_csv(DATA_FILE)

    # Drop ID columns. They are labels, not useful signals.
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # TotalCharges is sometimes text with blanks. Convert it to numbers.
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Turn Yes/No into 1/0.
    y = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[TARGET])

    # 80% for training, 20% kept hidden until the final test.
    # stratify keeps the same churn share in both parts.
    return train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )


def make_preprocessor(X):
    """Clean the inputs: fill gaps, scale numbers, turn text into 0/1 columns."""
    number_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    text_cols = [c for c in X.columns if c not in number_cols]

    numbers = Pipeline([
        ("fill_gaps", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    text = Pipeline([
        ("fill_gaps", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numbers", numbers, number_cols),
        ("text", text, text_cols),
    ])


def make_models(X):
    """The two models we compare. Each one includes the cleaning step."""
    preprocessor = make_preprocessor(X)
    return {
        "logistic_regression": Pipeline([
            ("clean", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("clean", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def pick_best_model(models, X_train, y_train):
    """Score each model with 5-fold cross-validation and return the best one."""
    scores = {}
    for name, model in models.items():
        # Trains on 4 parts, tests on the 5th, repeats 5 times, then averages.
        scores[name] = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        ).mean()
        print(f"{name:22s} ROC-AUC: {scores[name]:.3f}")

    best_name = max(scores, key=scores.get)
    print(f"Winner: {best_name}")
    return best_name, models[best_name], scores[best_name]


def pick_cutoff(model, X_train, y_train):
    """Find the best cut-off (between 0.1 and 0.9) for saying 'will churn'.
    Uses predictions the model made on data it had not trained on."""
    chances = cross_val_predict(
        model, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]

    cutoffs = np.linspace(0.10, 0.90, 81)
    f1_scores = [f1_score(y_train, chances >= c) for c in cutoffs]
    return float(cutoffs[int(np.argmax(f1_scores))])


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Compare the models and pick the winner.
    best_name, best_model, cv_score = pick_best_model(
        make_models(X_train), X_train, y_train
    )

    # Train the winner on all training data.
    best_model.fit(X_train, y_train)

    # Choose the cut-off.
    cutoff = pick_cutoff(best_model, X_train, y_train)
    print(f"Chosen cut-off: {cutoff:.2f}")

    # Final test. Done only once, on the hidden data.
    chances = best_model.predict_proba(X_test)[:, 1]
    predictions = (chances >= cutoff).astype(int)
    test_results = {
        "roc_auc": roc_auc_score(y_test, chances),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }
    print("Test results:", {k: round(v, 3) for k, v in test_results.items()})

    # Save the model (cleaning + model together) and a readable report.
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)

    report = {
        "best_model": best_name,
        "cross_validation_roc_auc": round(float(cv_score), 3),
        "cutoff": round(cutoff, 2),
        "test_results": {k: round(float(v), 3) for k, v in test_results.items()},
        "columns_needed": list(X_train.columns),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(f"Saved model to {MODEL_FILE} and report to {REPORT_FILE}")


if __name__ == "__main__":
    main()