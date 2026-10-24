import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


NUM_COLS = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

ALL_COLS = NUM_COLS + CAT_COLS


# ------------------ Input validation ------------------
def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = set(ALL_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    return df[ALL_COLS]


# ------------------ Pipelines ------------------
numeric_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, NUM_COLS),
        ('cat', categorical_pipeline, CAT_COLS)
    ]
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)


# ------------------ Train ------------------
def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_path: str = "artifacts/model.joblib"
):
    X_train = validate_input(X_train)
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, save_path)
    print(f"Model saved at: {save_path}")


# ------------------ Load ------------------
def load_pipeline(model_path: str = "artifacts/model.joblib"):
    return joblib.load(model_path)
