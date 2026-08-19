"""
src/train.py — Orchestrate data prep, training, and evaluation
"""

import json
from pathlib import Path

import structlog

from src.config import setup_logging, PIPELINE_PATH, RESULTS_PATH, MODEL_CONFIG
from src.data_prep import load_splits
from src.model import ChurnModel

log = structlog.get_logger(__name__)


def save_results(results: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("results_saved", path=str(path))


if __name__ == "__main__":
    setup_logging()

    X_train, X_test, y_train, y_test = load_splits()

    model = ChurnModel(
        estimator=MODEL_CONFIG["estimator"],
        params=MODEL_CONFIG["params"],
    )

    results = model.train(X_train, y_train)
    model.save(PIPELINE_PATH)
    save_results(results, RESULTS_PATH)
