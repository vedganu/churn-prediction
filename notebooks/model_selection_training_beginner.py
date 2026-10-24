"""
Telecom Churn - Model Selection and Training (beginner-friendly version)

WHAT THIS SCRIPT DOES
---------------------
It tries several machine learning models, picks the best one, and saves it.

   1. Makes sure the training and test data files exist.
   2. Tries 4 models (a simple "baseline", logistic regression,
      decision tree and random forest) with different settings.
   3. Scores each one with cross-validation (training data only).
   4. Picks the best model.
   5. Picks the best "cut-off" for deciding who is likely to churn.
   6. Checks the final model on the test data - only once.
   7. Saves the model and a report.

HOW TO RUN IT
-------------
    python model_selection_training_beginner.py train --project-dir path/to/project

   Optional settings:
       --seed 42          (random number seed, keeps results the same every run)
       --cv-folds 5       (how many parts the training data is split into)
       --test-size 0.20   (fraction of data kept aside for testing)

FILES IT NEEDS (inside the project folder)
------------------------------------------
    data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
        (created automatically from data/processed/cleaned_churn.csv if missing)
    models/preprocessor.pkl
        (the data-cleaning recipe made in the preprocessing step)

FILES IT CREATES
----------------
    models/churn_pipeline.pkl       -> the final model (ready to use)
    reports/model_comparison.csv    -> table comparing all models
    reports/model_metadata.json     -> settings, scores and the chosen cut-off
"""

# ---------------------------------------------------------------------------
# Part 1: Imports - bringing in the tools we need
# ---------------------------------------------------------------------------
# Why: we do not write these tools ourselves. Libraries like pandas (tables),
# scikit-learn (machine learning) and joblib (saving files) do the heavy work.

from __future__ import annotations

import argparse        # reads the options typed after the command
import json            # writes the report file
import logging         # prints clear progress messages
import platform        # records the Python version used
import sys
import time            # measures how long each model takes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib          # saves and loads the model file
import numpy as np
import pandas as pd    # handles tables of data
import sklearn

from sklearn.base import clone                       # makes a fresh, unused copy of a recipe
from sklearn.dummy import DummyClassifier            # the simple "baseline" model
from sklearn.ensemble import RandomForestClassifier  # a model made of many decision trees
from sklearn.linear_model import LogisticRegression  # a simple, fast model
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,        # tries every combination of settings
    StratifiedKFold,     # splits the data into parts, keeping the churn ratio the same in each
    cross_val_predict,
    train_test_split,    # splits the data into training and test sets
)
from sklearn.pipeline import Pipeline                # chains cleaning + model into one object
from sklearn.tree import DecisionTreeClassifier      # a model that asks yes/no questions

# A logger prints messages like "12:00:01 INFO saved model".
log = logging.getLogger("churn.train")

# Column name of the answer we want to predict (1 = customer left, 0 = stayed).
TARGET = "Churn"

# Names of the saved files.
PIPELINE_FILE = "churn_pipeline.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"


# ---------------------------------------------------------------------------
# Part 2: Settings - all important numbers in one place
# ---------------------------------------------------------------------------
# Why: if we want to change the random seed or the folder, we change it here
# and nowhere else. This makes results easy to repeat.

@dataclass(frozen=True)
class Config:
    project_dir: Path = Path(".")     # the main project folder
    random_state: int = 42            # the random seed: same seed = same results
    test_size: float = 0.20           # 20% of data kept for the final test
    cv_folds: int = 5                 # cross-validation splits the training data into 5 parts
    primary_metric: str = "roc_auc"   # the score used to pick the winner

    # Folders are built from the project folder. Read them as "project/data/processed" etc.
    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data" / "processed"

    @property
    def model_dir(self) -> Path:
        return self.project_dir / "models"

    @property
    def report_dir(self) -> Path:
        return self.project_dir / "reports"


# ---------------------------------------------------------------------------
# Part 3: Make sure the training and test files exist
# ---------------------------------------------------------------------------

def make_split_files_if_missing(cfg: Config) -> None:
    """Create the train/test files from the cleaned data, if they are not there.

    WHY: the model needs the data split into a training part and a test part.
    The test part must be kept hidden until the very end, so we make sure
    the split exists and matches the one used in the preprocessing step.
    """
    file_names = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

    # If all four files already exist, there is nothing to do.
    if all((cfg.data_dir / name).exists() for name in file_names):
        return

    cleaned_file = cfg.data_dir / "cleaned_churn.csv"
    if not cleaned_file.exists():
        raise FileNotFoundError(
            f"Cannot find {cleaned_file}. Run the preprocessing step first."
        )

    # Load the cleaned data.
    data = pd.read_csv(cleaned_file)

    # The answer column may still say "Yes"/"No". The model needs 1 and 0.
    if data[TARGET].dtype == object:
        data[TARGET] = data[TARGET].map({"Yes": 1, "No": 0})

    # X = the customer details (inputs), y = the answer (churn yes/no).
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    # Split into training (80%) and test (20%) parts.
    # "stratify=y" keeps the same share of churners in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Safety check: if the preprocessing step already saved a y_train file,
    # our new split must be exactly the same. Otherwise the recipe and
    # the data would not match.
    old_file = cfg.data_dir / "y_train.csv"
    if old_file.exists():
        old_y_train = pd.read_csv(old_file).squeeze("columns")
        if not np.array_equal(old_y_train.values, y_train.values):
            raise ValueError(
                "The new split does not match the preprocessing step's split. "
                "Check test_size, random_state and stratify."
            )

    # Save the four files.
    X_train.to_csv(cfg.data_dir / "X_train.csv", index=False)
    X_test.to_csv(cfg.data_dir / "X_test.csv", index=False)
    y_train.to_csv(cfg.data_dir / "y_train.csv", index=False)
    y_test.to_csv(cfg.data_dir / "y_test.csv", index=False)
    log.info("Created train/test files in %s", cfg.data_dir)


# ---------------------------------------------------------------------------
# Part 4: Load the data and check it is correct
# ---------------------------------------------------------------------------

def load_data(cfg: Config):
    """Load the four data files and check that they look right.

    WHY: if the data is wrong, every result after this would be wrong too.
    Checking early stops the script with a clear message instead of
    giving misleading results.
    """
    X_train = pd.read_csv(cfg.data_dir / "X_train.csv")
    X_test = pd.read_csv(cfg.data_dir / "X_test.csv")
    y_train = pd.read_csv(cfg.data_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(cfg.data_dir / "y_test.csv").squeeze("columns")

    # Check 1: each row of inputs has exactly one answer.
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Number of rows in inputs and answers do not match.")

    # Check 2: training and test data have the same columns in the same order.
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test data have different columns.")

    # Check 3: the answer must only be 0 or 1.
    if not set(y_train.unique()) <= {0, 1}:
        raise ValueError("The answer column must only contain 0 or 1.")

    # Check 4: the data must still have text columns (not yet converted to numbers).
    # The cleaning recipe does that conversion itself, so it must receive raw data.
    if X_train.select_dtypes(exclude="number").shape[1] == 0:
        raise ValueError("Expected raw data with text columns, but found only numbers.")

    log.info(
        "Training rows: %d (churn rate %.1f%%) | Test rows: %d (churn rate %.1f%%)",
        len(X_train), 100 * y_train.mean(), len(X_test), 100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


def load_cleaning_recipe(cfg: Config):
    """Load the cleaning recipe from the preprocessing step.

    WHY clone(): the saved recipe has already learned from ALL the training
    data. We take a fresh copy with the rules but without what it learned,
    so each check during cross-validation learns only from its own part of the data.
    This prevents "cheating" by letting the check see its own answers.
    """
    recipe_file = cfg.model_dir / PREPROCESSOR_FILE
    if not recipe_file.exists():
        raise FileNotFoundError(f"Cannot find {recipe_file}. Run the preprocessing step first.")
    return clone(joblib.load(recipe_file))


# ---------------------------------------------------------------------------
# Part 5: Define the candidate models
# ---------------------------------------------------------------------------

def build_candidates(recipe, seed: int) -> dict:
    """Prepare the list of models to compare. Nothing is trained here yet.

    WHY a Pipeline: each model is glued to the same cleaning recipe. Then the
    cleaning and the model always go together, so no step can be forgotten
    when we use the model later.

    WHY a grid: a "setting" such as tree depth is chosen before training.
    The grid lists the values to try, and GridSearchCV tests every combination.
    """

    def make_pipeline(model):
        # Cleaning recipe first (fresh copy), then the model.
        return Pipeline([("prep", clone(recipe)), ("model", model)])

    return {
        # 1. Baseline: always guesses the average churn rate.
        #    WHY: any real model must do better than this, or it is useless.
        "baseline": (make_pipeline(DummyClassifier(strategy="prior")), {}),

        # 2. Logistic regression: simple and fast.
        #    WHY class_weight="balanced": only about 27% of customers churn,
        #    so this tells the model that missing a churner matters more.
        "logreg": (
            make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
            {"model__C": [0.01, 0.1, 1, 10]},   # 4 settings to try
        ),

        # 3. Decision tree: a tree of yes/no questions.
        #    max_depth = how many questions deep; min_samples_leaf = smallest group allowed.
        "decision_tree": (
            make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
            {
                "model__max_depth": [3, 5, 8],               # 3 options
                "model__min_samples_leaf": [5, 20, 50],      # 3 options -> 9 combinations
            },
        ),

        # 4. Random forest: many decision trees voting together.
        #    n_jobs=1 because the grid search already uses all CPU cores.
        "random_forest": (
            make_pipeline(
                RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1)
            ),
            {
                "model__n_estimators": [300],                 # number of trees
                "model__max_depth": [6, 10, None],            # 3 options
                "model__min_samples_leaf": [1, 5, 10],        # 3 options -> 9 combinations
            },
        ),
    }


# ---------------------------------------------------------------------------
# Part 6: Train and compare every model fairly
# ---------------------------------------------------------------------------

# The scores we record for every model.
SCORING = {
    "roc_auc": "roc_auc",                # overall ability to rank churners above non-churners
    "average_precision": "average_precision",
    "recall": make_scorer(recall_score, zero_division=0),        # how many churners we catch
    "precision": make_scorer(precision_score, zero_division=0),  # how many flagged are real churners
    "f1": make_scorer(f1_score, zero_division=0),                # balance of precision and recall
}


def tune_and_compare(candidates: dict, X_train, y_train, cfg: Config):
    """Train every model with the same cross-validation, then rank them.

    WHY cross-validation: instead of one split, we split the training data
    into 5 parts, train on 4 and test on the 5th, and repeat 5 times.
    The average score is more reliable than a single split.

    WHY the same folds for everyone: every model gets exactly the same
    test parts, so the comparison is fair.
    """
    # The fold splitter. shuffle=True mixes the rows; random_state keeps it repeatable.
    folds = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    searches = {}   # keeps the full search result for each model
    rows = []       # one summary row per model, for the comparison table

    for name, (pipeline, grid) in candidates.items():
        start = time.time()

        # GridSearchCV: try every setting in the grid with cross-validation.
        # refit=primary_metric: after searching, retrain the best version on ALL training data.
        search = GridSearchCV(
            pipeline, grid, cv=folds, scoring=SCORING,
            refit=cfg.primary_metric, n_jobs=-1,
        )
        search.fit(X_train, y_train)

        # Pull out the scores of the best setting found for this model.
        best = search.best_index_
        row = {"model": name}
        for metric in SCORING:
            row[metric] = search.cv_results_[f"mean_test_{metric}"][best]
        row[f"{cfg.primary_metric}_std"] = search.cv_results_[f"std_test_{cfg.primary_metric}"][best]
        row["best_params"] = search.best_params_
        row["seconds"] = round(time.time() - start, 1)

        rows.append(row)
        searches[name] = search
        log.info("%-14s score (%s) = %.3f", name, cfg.primary_metric, row[cfg.primary_metric])

    # Sort so the best model is at the top.
    table = pd.DataFrame(rows).sort_values(cfg.primary_metric, ascending=False)
    table = table.reset_index(drop=True)
    return searches, table


# ---------------------------------------------------------------------------
# Part 7: Pick the cut-off for "will churn"
# ---------------------------------------------------------------------------

def choose_threshold(best_pipeline, X_train, y_train, cfg: Config) -> float:
    """Find the best cut-off between 0 and 1 for saying "this customer will churn".

    WHY not just 0.5: the model gives a chance (probability) for each customer.
    With rare churners, 0.5 is usually not the best place to draw the line.

    WHY out-of-fold predictions: each customer's chance is predicted by a model
    that never saw that customer. This keeps the choice honest.
    """
    folds = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    # Get a churn chance for every training customer, from a model that never saw them.
    chances = cross_val_predict(
        clone(best_pipeline), X_train, y_train,
        cv=folds, method="predict_proba", n_jobs=-1,
    )[:, 1]

    # Try cut-offs from 0.10 to 0.90 and keep the one with the best F1 score.
    possible_cutoffs = np.linspace(0.10, 0.90, 81)
    f1_scores = [f1_score(y_train, chances >= cutoff) for cutoff in possible_cutoffs]
    return float(possible_cutoffs[int(np.argmax(f1_scores))])


# ---------------------------------------------------------------------------
# Part 8: Final check on the test data (only once)
# ---------------------------------------------------------------------------

def evaluate_on_test(pipeline, threshold: float, X_test, y_test) -> dict:
    """Score the final model on the held-out test data.

    WHY only once: the test data is a fair exam. If we looked at the score
    and changed the model, the exam would no longer be fair.
    """
    chances = pipeline.predict_proba(X_test)[:, 1]
    predictions = (chances >= threshold).astype(int)

    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_test, chances)),
        "average_precision": float(average_precision_score(y_test, chances)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(y_test, predictions).tolist(),
    }


# ---------------------------------------------------------------------------
# Part 9: Save the model and its report
# ---------------------------------------------------------------------------

def save_results(best_name, pipeline, threshold, best_row, test_metrics,
                 comparison, X_train, cfg: Config) -> None:
    """Save the model file and the report files.

    WHY save a report: months later, you can see exactly which model was
    chosen, which settings were used and how well it did.
    """
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    # The model file. It contains the cleaning steps AND the model together.
    joblib.dump(pipeline, cfg.model_dir / PIPELINE_FILE)

    # The table comparing all models.
    comparison.to_csv(cfg.report_dir / "model_comparison.csv", index=False)

    # Summary of everything important about this training run.
    report = {
        "model": best_name,
        "best_params": best_row["best_params"],
        "decision_threshold": threshold,
        "cv_metrics": {metric: best_row[metric] for metric in SCORING},
        "test_metrics": test_metrics,
        "n_train_rows": len(X_train),
        # The columns the model expects as input. Any app using the model must provide these.
        "input_columns": list(X_train.columns),
        "random_state": cfg.random_state,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    (cfg.report_dir / "model_metadata.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    log.info("Saved model to %s and reports to %s", cfg.model_dir, cfg.report_dir)


# ---------------------------------------------------------------------------
# Part 10: The main training steps, in order
# ---------------------------------------------------------------------------

def train(cfg: Config) -> dict:
    """Run every step in order: prepare, compare, choose, test, save."""

    # Step A: make sure the files exist.
    make_split_files_if_missing(cfg)

    # Step B: load the data and the cleaning recipe.
    X_train, X_test, y_train, y_test = load_data(cfg)
    recipe = load_cleaning_recipe(cfg)

    # Step C: compare all models with cross-validation.
    candidates = build_candidates(recipe, cfg.random_state)
    searches, comparison = tune_and_compare(candidates, X_train, y_train, cfg)

    # Step D: the top row of the table is the winner.
    best_row = comparison.iloc[0].to_dict()
    best_name = best_row["model"]
    # This model was already retrained on all the training data by GridSearchCV.
    best_pipeline = searches[best_name].best_estimator_
    log.info("Winner: %s with settings %s", best_name, best_row["best_params"])

    # Step E: choose the cut-off for "will churn".
    threshold = choose_threshold(best_pipeline, X_train, y_train, cfg)
    log.info("Chosen cut-off: %.2f", threshold)

    # Step F: final test, done only once.
    test_metrics = evaluate_on_test(best_pipeline, threshold, X_test, y_test)
    log.info(
        "Test results: ROC-AUC=%.3f, precision=%.3f, recall=%.3f, F1=%.3f",
        test_metrics["roc_auc"], test_metrics["precision"],
        test_metrics["recall"], test_metrics["f1"],
    )

    # Step G: save everything.
    save_results(best_name, best_pipeline, threshold, best_row,
                 test_metrics, comparison, X_train, cfg)

    # Step H: quick check that the saved file loads and works.
    loaded_model = joblib.load(cfg.model_dir / PIPELINE_FILE)
    loaded_model.predict_proba(X_test.head(3))
    log.info("Check passed: the saved model loads and runs.")

    return test_metrics


# ---------------------------------------------------------------------------
# Part 11: Read the command you typed and start the work
# ---------------------------------------------------------------------------

def parse_arguments(argv):
    """Read the options typed after the script name."""
    parser = argparse.ArgumentParser(description="Train and select the churn model.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_cmd = subcommands.add_parser("train", help="Train, compare and save the model.")
    train_cmd.add_argument("--project-dir", type=Path, default=Path("."),
                           help="The main project folder.")
    train_cmd.add_argument("--seed", type=int, default=42,
                           help="Random seed, so results can be repeated.")
    train_cmd.add_argument("--cv-folds", type=int, default=5,
                           help="Number of cross-validation parts.")
    train_cmd.add_argument("--test-size", type=float, default=0.20,
                           help="Fraction of data kept for the final test.")
    train_cmd.add_argument("--log-level", default="INFO",
                           help="How much to print: INFO, WARNING or ERROR.")

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)

    # Set up the progress messages.
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "train":
        cfg = Config(
            project_dir=args.project_dir.resolve(),
            random_state=args.seed,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
        )
        train(cfg)

    return 0


# This line means: "run main() only when this file is started directly".
# It does not run when another program imports this file.
if __name__ == "__main__":
    raise SystemExit(main())
