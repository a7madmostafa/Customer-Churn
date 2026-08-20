"""
src/config.py — Centralized configuration for the project
"""

import logging
from pathlib import Path

import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def setup_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ])

RANDOM_STATE = 42
N_TRIALS = 50

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "telecom_churn.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODELS_DIR = BASE_DIR / "models"
ONNX_PATH = MODELS_DIR / "pipeline.onnx"
RESULTS_PATH = MODELS_DIR / "results.json"

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

MODEL_REGISTRY = {
    "logistic_regression": {
        "class": LogisticRegression,
        "fixed_params": {"max_iter": 2000, "random_state": RANDOM_STATE},
        "search_space": {
            "C": ("float", 0.01, 100.0, "log"),
            "solver": ("categorical", ["liblinear", "lbfgs"]),
            "class_weight": ("categorical", [None, "balanced"]),
        },
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "fixed_params": {"random_state": RANDOM_STATE},
        "search_space": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 3, 20),
            "min_samples_split": ("int", 2, 10),
            "min_samples_leaf": ("int", 1, 10),
            "class_weight": ("categorical", [None, "balanced"]),
        },
    },
    "xgboost": {
        "class": XGBClassifier,
        "fixed_params": {"random_state": RANDOM_STATE, "eval_metric": "logloss"},
        "search_space": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3, "log"),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "scale_pos_weight": ("float", 1.0, 5.0),
        },
    },
}
