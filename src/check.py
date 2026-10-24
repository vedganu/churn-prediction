import os

for f in ["clean.py", "preprocessor.py", "train_model.py", "churn_model.pkl"]:
    print(f, os.path.exists(f))