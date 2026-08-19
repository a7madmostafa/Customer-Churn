"""
src/config.py — Centralized configuration for the project
"""

import logging
from pathlib import Path
from typing import Any, Dict

import structlog
from sklearn.linear_model import LogisticRegression


def setup_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ])

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "telecom_churn.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
RESULTS_PATH = MODELS_DIR / "results.json"

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "params": {
        "model__C": [0.1, 1.0, 10.0],
        "model__solver": ["liblinear"],
        "model__class_weight": [None, "balanced"],
    },
}
