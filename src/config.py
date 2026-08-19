"""
src/config.py — Centralized configuration for the project
"""

from pathlib import Path
from typing import Any, Dict
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telecom_churn.csv"
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
