"""
src/evaluate.py — Load trained model and evaluate on test set
"""

import argparse

import structlog

from src.config import setup_logging, PIPELINE_PATH
from src.data_prep import load_splits
from src.model import ChurnModel

log = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained churn model")
    parser.add_argument("--model", type=str, default=str(PIPELINE_PATH), help="Path to trained pipeline")
    args = parser.parse_args()

    setup_logging()

    _, X_test, _, y_test = load_splits()

    model = ChurnModel(estimator=None, params={})
    model.load(args.model)

    metrics = model.evaluate(X_test, y_test)
    log.info("test_metrics", accuracy=metrics["accuracy"], roc_auc=metrics["roc_auc"])


if __name__ == "__main__":
    main()
