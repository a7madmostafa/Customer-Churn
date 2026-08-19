# Customer Churn Prediction

A modular ML pipeline for predicting customer churn in telecommunications, built with scikit-learn, FastAPI, and structlog.

## Project Structure

```
S1_Project/
├── pyproject.toml              # Dependencies and build config (uv)
├── data/
│   ├── telecom_churn.csv       # Raw dataset (7,043 customers)
│   ├── train.csv               # Training split (5,634 rows)
│   └── test.csv                # Test split (1,409 rows)
├── models/
│   ├── pipeline.pkl            # Trained sklearn pipeline
│   └── results.json            # CV results
└── src/
    ├── __init__.py             # Package exports
    ├── config.py               # Paths, constants, logging setup
    ├── data_prep.py            # Load, clean, split, save data
    ├── model.py                # ChurnModel class (train, predict, evaluate, save, load)
    ├── schemas.py              # Pydantic request/response models
    ├── train.py                # Training orchestrator
    ├── evaluate.py             # Model evaluation on test set
    ├── inference.py            # CLI batch prediction
    └── app.py                  # FastAPI serving endpoint
```

## Quickstart

### 1. Install dependencies

```bash
uv sync
```

### 2. Prepare data

```bash
uv run python -m src.data_prep
```

Loads raw CSV, cleans it, splits 80/20 stratified, saves `data/train.csv` and `data/test.csv`.

### 3. Train model

```bash
uv run python -m src.train
```

Trains LogisticRegression with GridSearchCV (3-fold stratified CV, ROC-AUC scoring), saves best pipeline to `models/pipeline.pkl`.

### 4. Evaluate

```bash
uv run python -m src.evaluate
```

### 5. Serve API

```bash
uv run uvicorn src.app:app --port 8000
```

Interactive docs at http://localhost:8000/docs

## API Endpoints

| Method | Path       | Description            |
|--------|------------|------------------------|
| GET    | `/health`  | Model status check     |
| POST   | `/predict` | Single customer prediction |

### Predict Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
```

### Predict Response

```json
{
  "churn": 1,
  "probability": 0.7381
}
```

## Batch Inference

```bash
uv run python -m src.inference --input new_customers.csv --output predictions.csv
```

## Results

| Metric   | Score  |
|----------|--------|
| ROC-AUC  | 0.8415 |
| Accuracy | 0.8055 |

Best model: **LogisticRegression** (C=10.0, liblinear, no class weights)

## Tech Stack

- **Python** >= 3.11
- **scikit-learn** — model training and pipeline
- **FastAPI** + **uvicorn** — API serving
- **structlog** — structured JSON logging
- **pydantic** — request/response validation
- **uv** — dependency management
