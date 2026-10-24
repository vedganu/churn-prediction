import pandas as pd
from src.pipeline import load_pipeline,validate_input

MODEL_PATH  = "artifacts/model.joblib"


def predict(df:pd.DataFrame):
    """
    Takes raw input dataframe and return predictions.
    """

    model = load_pipeline(MODEL_PATH)

    # validate & fix column order
    df = validate_input(df)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:,1]

    return preds,probs



def main():

    sample_data = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.5,

        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }

    df = pd.DataFrame([sample_data])

    preds,probs = predict(df)

    print("Prediction (0 = No churn, 1=Churn):",preds[0])
    print("Churn probability :", round(probs[0],3))

if __name__ == "__main__":
    main()


    