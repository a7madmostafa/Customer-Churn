"""
src/inference.py — Load trained model and predict on new data
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import structlog

from src.config import setup_logging, PIPELINE_PATH
from src.model import ChurnModel

log = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument("--model", type=str, default=str(PIPELINE_PATH), help="Path to trained pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV with new customers")
    parser.add_argument("--output", type=str, default=None, help="Path to save predictions (optional)")
    args = parser.parse_args()

    setup_logging()

    model = ChurnModel(estimator=None, params={})
    model.load(args.model)

    df = pd.read_csv(args.input)
    log.info("input_loaded", rows=len(df))

    predictions = model.predict(df)
    df["churn_prediction"] = predictions

    print(df[["churn_prediction"]].to_string())

    if args.output:
        df.to_csv(args.output, index=False)
        log.info("predictions_saved", path=args.output, rows=len(df))


if __name__ == "__main__":
    main()
