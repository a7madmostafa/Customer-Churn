"""
src/inference.py — Load trained model and predict on new data
"""

import argparse

import pandas as pd
import structlog

from src.config import setup_logging, ONNX_PATH
from src.model import ChurnModel

log = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument("--model", type=str, default=str(ONNX_PATH), help="Path to ONNX model")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV with new customers")
    parser.add_argument("--output", type=str, default=None, help="Path to save predictions (optional)")
    args = parser.parse_args()

    setup_logging()

    model = ChurnModel()
    model.load_onnx(args.model)

    df = pd.read_csv(args.input)
    log.info("input_loaded", rows=len(df))

    classes, _ = model.predict_onnx(df)
    df["churn_prediction"] = classes

    print(df[["churn_prediction"]].to_string())

    if args.output:
        df.to_csv(args.output, index=False)
        log.info("predictions_saved", path=args.output, rows=len(df))


if __name__ == "__main__":
    main()
