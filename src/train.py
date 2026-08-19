"""
src/train.py — Train, tune, compare models, and save the best pipeline
"""

import os
import pickle
import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, NUM_COLS, CAT_COLS, MODEL_CONFIG, DATA_PATH, PIPELINE_PATH, RESULTS_PATH
from src.data_prep import prepare_data

def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for use inside a Pipeline."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
    ])

def build_pipeline(preprocessor, estimator) -> Pipeline:
    """Create a full sklearn Pipeline: preprocessing + model."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])



def train_and_tune(
    model_config: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "roc_auc",
    cv: int = 3,
    n_jobs: int = -1,
    save_path: str = "models/pipeline.pkl",
) -> Tuple[Pipeline, List[Dict[str, Any]]]:
    """
    Train and tune multiple models, select the best one, and save it.

    Args:
        X_train:   Raw training features.
        y_train:   Training target.
        scoring:   Metric used by GridSearchCV to select the best model.
                   Good options: "roc_auc" (default), "f1", "recall".
        cv:        Number of StratifiedKFold folds.
        n_jobs:    Parallel jobs (-1 = all cores, 1 = stable/single-core).
        save_path: Where to save the selected fitted pipeline.

    Returns:
        best_pipeline: The fitted best sklearn Pipeline.
        results_df:    Model comparison table sorted by CV score descending.
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    grid_params = model_config["params"]
    estimator = model_config["estimator"]
    preprocessor = build_preprocessor()
    pipeline = build_pipeline(preprocessor, estimator)
    grid = GridSearchCV(pipeline, grid_params, cv=kf, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    print(f"  Best {scoring} score: {grid.best_score_:.3f}")
    results_df = pd.DataFrame(grid.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    results = results_df.to_dict(orient="records")
    return best_pipeline, results
    
def save_model(pipeline: Pipeline, path: str = "models/pipeline.pkl") -> None:
    """Save the fitted best pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  Best pipeline saved to: {path}")
    
def save_results(results: List[Dict[str, Any]], path: str = "models/results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    best_pipeline, results = train_and_tune(MODEL_CONFIG, X_train, y_train)
    save_model(best_pipeline, PIPELINE_PATH)
    save_results(results, RESULTS_PATH)
    
