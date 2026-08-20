"""
src/app.py — FastAPI serving endpoint
"""

import numpy as np
import pandas as pd
import structlog
from fastapi import FastAPI

from src.config import ONNX_PATH
from src.model import ChurnModel
from src.schemas import ChurnRequest, ChurnResponse

log = structlog.get_logger(__name__)

app = FastAPI(title="Customer Churn Prediction API")

model = ChurnModel()


@app.on_event("startup")
def load_model() -> None:
    model.load_onnx(ONNX_PATH)
    log.info("onnx_loaded", path=str(ONNX_PATH))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model.onnx_session is not None}


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest) -> ChurnResponse:
    df = pd.DataFrame([req.model_dump()])
    classes, probabilities = model.predict_onnx(df)
    prediction = int(classes[0])
    probability = float(probabilities[0][1])
    log.info("prediction_done", churn=prediction, probability=f"{probability:.3f}")
    return ChurnResponse(churn=prediction, probability=round(probability, 4))
