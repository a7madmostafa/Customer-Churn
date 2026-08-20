"""
src/train.py — Orchestrate data prep, training, and evaluation
"""

import json
from pathlib import Path

import mlflow
import optuna
import structlog

from src.config import setup_logging, ONNX_PATH, RESULTS_PATH, N_TRIALS
from src.data_prep import load_splits
from src.model import ChurnModel

log = structlog.get_logger(__name__)


def objective(trial, X_train, y_train, model):
    """Optuna objective function — returns CV ROC-AUC score."""
    return model.train_with_optuna(trial, X_train, y_train)


if __name__ == "__main__":
    setup_logging()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mlflow.set_experiment("churn_prediction")

    X_train, X_test, y_train, y_test = load_splits()

    model = ChurnModel()

    study = optuna.create_study(direction="maximize", study_name="churn_prediction")
    study.optimize(lambda trial: objective(trial, X_train, y_train, model), n_trials=N_TRIALS)

    best = study.best_trial
    log.info("study_done", best_value=f"{best.value:.4f}", best_params=best.params)

    best_model_name = best.params["model_name"]
    best_model_params = {k: v for k, v in best.params.items() if k != "model_name"}

    model.build_best_model(best_model_name, best_model_params, X_train, y_train)
    model.export_onnx(ONNX_PATH)

    with mlflow.start_run(run_name=best_model_name):
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_params(best_model_params)
        mlflow.log_metric("cv_roc_auc", best.value)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_artifact(str(ONNX_PATH))
        log.info("mlflow_run_logged", model=best_model_name, cv_roc_auc=f"{best.value:.4f}")

    results = {
        "best_model": best_model_name,
        "best_params": best.params,
        "best_cv_roc_auc": best.value,
        "n_trials": N_TRIALS,
    }
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("results_saved", path=str(RESULTS_PATH))
