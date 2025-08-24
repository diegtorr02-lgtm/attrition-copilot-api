from __future__ import annotations
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os, pandas as pd

from attrition_copilot import load_bundle, predict_for_employee, explain_instance

API_KEY = os.getenv("ATTRITION_API_KEY", "dev-secret")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

app = FastAPI(title="Attrition Copilot API", version="1.0.0")

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # für Tests alle Domains zulassen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = load_bundle(MODEL_PATH)

def require_api_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")
    return True

class PredictRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest):
    payload = {k: v for k, v in req.features.items() if v is not None}
    if len(payload) < 5:
        raise HTTPException(400, "Bitte mindestens ~5 Kernfelder angeben (z.B. Age, MonthlyIncome, OverTime, DistanceFromHome, BusinessTravel).")

    df = pd.DataFrame([payload])
    if "Attrition" not in df.columns:
        df["Attrition"] = 0
    if "EmployeeNumber" not in df.columns:
        df["EmployeeNumber"] = "api-0"

    proba, _ = predict_for_employee(_bundle, df, "api-0")
    expl = explain_instance(_bundle, df, "api-0", topk=3)

    return {
        "probability": round(float(proba), 4),
        "risk_label": "high" if proba >= 0.5 else "low",
        "explain_method": expl.get("explain_method"),
        "top_reasons": expl.get("top_reasons"),
    }
