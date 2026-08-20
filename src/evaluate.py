"""
src/evaluate.py — Load trained model and evaluate on test set
"""

import argparse

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import setup_logging, ONNX_PATH
from src.data_prep import load_splits
from src.model import ChurnModel

log = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained churn model")
    parser.add_argument("--model", type=str, default=str(ONNX_PATH), help="Path to ONNX model")
    args = parser.parse_args()

    setup_logging()

    _, X_test, _, y_test = load_splits()

    model = ChurnModel()
    model.load_onnx(args.model)

    classes, probabilities = model.predict_onnx(X_test)
    y_pred = classes
    y_proba = probabilities[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    log.info("test_metrics", accuracy=metrics["accuracy"], roc_auc=metrics["roc_auc"])


if __name__ == "__main__":
    main()
