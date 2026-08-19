"""
src/model.py — ChurnModel class wrapping the sklearn pipeline
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import structlog
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import RANDOM_STATE, NUM_COLS, CAT_COLS

log = structlog.get_logger(__name__)


class ChurnModel:
    """Wraps preprocessing + estimator into a single trainable pipeline."""

    def __init__(self, estimator: Any, params: Dict[str, Any]) -> None:
        self.estimator = estimator
        self.params = params
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
        ])
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.estimator),
        ])

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scoring: str = "roc_auc",
        cv: int = 3,
        n_jobs: int = -1,
    ) -> List[Dict[str, Any]]:
        """Train with GridSearchCV, return sorted results."""
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        pipeline = self._build_pipeline()
        grid = GridSearchCV(pipeline, self.params, cv=kf, scoring=scoring, n_jobs=n_jobs)
        grid.fit(X_train, y_train)

        self.pipeline = grid.best_estimator_
        log.info("training_done", scoring=scoring, best_score=f"{grid.best_score_:.3f}")

        results_df = pd.DataFrame(grid.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score")
        return results_df.to_dict(orient="records")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict on new data."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet. Call train() or load() first.")
        return self.pipeline.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate on test set, return metrics dict."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet. Call train() or load() first.")

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        log.info("evaluation_done", accuracy=metrics["accuracy"], roc_auc=metrics["roc_auc"])
        print(classification_report(y_test, y_pred))
        return metrics

    def save(self, path: str | Path) -> None:
        """Save fitted pipeline to disk."""
        if self.pipeline is None:
            raise RuntimeError("No pipeline to save. Call train() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        log.info("model_saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Load fitted pipeline from disk."""
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        log.info("model_loaded", path=str(path))
