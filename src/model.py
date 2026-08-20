"""
src/model.py — ChurnModel class wrapping the sklearn pipeline
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
import pandas as pd
import structlog
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import RANDOM_STATE, NUM_COLS, CAT_COLS, MODEL_REGISTRY

log = structlog.get_logger(__name__)


def suggest_params(trial, model_name: str) -> Dict[str, Any]:
    """Optuna suggests hyperparameters based on search_space from config."""
    params = {}
    for name, spec in MODEL_REGISTRY[model_name]["search_space"].items():
        kind = spec[0]
        if kind == "float":
            log_scale = len(spec) > 3 and spec[3] == "log"
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=log_scale)
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


class ChurnModel:
    """Wraps preprocessing + estimator into a single trainable pipeline."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.onnx_session: ort.InferenceSession | None = None

    def _build_pipeline(self, model_name: str, model_params: Dict[str, Any]) -> Pipeline:
        """Build pipeline with given model name and hyperparams."""
        config = MODEL_REGISTRY[model_name]
        estimator = config["class"](**config["fixed_params"], **model_params)

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
        ])

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

    def train_with_optuna(
        self,
        trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 3,
    ) -> float:
        """Train with Optuna trial, return CV ROC-AUC score."""
        model_name = trial.suggest_categorical("model_name", list(MODEL_REGISTRY.keys()))
        model_params = suggest_params(trial, model_name)

        pipeline = self._build_pipeline(model_name, model_params)

        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="roc_auc", n_jobs=-1)

        return scores.mean()

    def build_best_model(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Build and fit the best model from Optuna study."""
        pipeline = self._build_pipeline(model_name, model_params)
        pipeline.fit(X_train, y_train)
        self.pipeline = pipeline
        log.info("best_model_trained", model=model_name, params=model_params)

    def export_onnx(self, path: str | Path) -> None:
        """Export full pipeline (preprocessing + model) to ONNX format."""
        if self.pipeline is None:
            raise RuntimeError("No pipeline to export.")

        initial_types = []
        for col in NUM_COLS:
            initial_types.append((col, FloatTensorType([None, 1])))
        for col in CAT_COLS:
            initial_types.append((col, StringTensorType([None, 1])))

        onnx_model = convert_sklearn(self.pipeline, initial_types=initial_types)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        log.info("onnx_exported", path=str(path))

    def load_onnx(self, path: str | Path) -> None:
        """Load ONNX model for inference."""
        self.onnx_session = ort.InferenceSession(str(path))
        log.info("onnx_loaded", path=str(path))

    def predict_onnx(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict using ONNX runtime. Returns (classes, probabilities)."""
        if self.onnx_session is None:
            raise RuntimeError("ONNX model not loaded. Call load_onnx() first.")

        input_dict = {}
        for col in NUM_COLS:
            input_dict[col] = df[col].values.astype(np.float32).reshape(-1, 1)
        for col in CAT_COLS:
            input_dict[col] = df[col].values.astype(object).reshape(-1, 1)

        classes, prob_dicts = self.onnx_session.run(None, input_dict)
        proba = np.array([[v for v in d.values()] for d in prob_dicts])
        return classes, proba
