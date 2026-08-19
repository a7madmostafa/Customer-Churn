"""
src/app.py — FastAPI serving endpoint
"""

import pandas as pd
import structlog
from fastapi import FastAPI

from src.config import PIPELINE_PATH
from src.model import ChurnModel
from src.schemas import ChurnRequest, ChurnResponse

log = structlog.get_logger(__name__)

app = FastAPI(title="Customer Churn Prediction API")

model = ChurnModel(estimator=None, params={})


@app.on_event("startup")
def load_model() -> None:
    model.load(PIPELINE_PATH)
    log.info("model_loaded", path=str(PIPELINE_PATH))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model.pipeline is not None}


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest) -> ChurnResponse:
    df = pd.DataFrame([req.model_dump()])
    prediction = int(model.predict(df)[0])
    probability = float(model.pipeline.predict_proba(df)[0][1])
    log.info("prediction_done", churn=prediction, probability=f"{probability:.3f}")
    return ChurnResponse(churn=prediction, probability=round(probability, 4))
