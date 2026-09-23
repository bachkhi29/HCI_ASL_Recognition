"""One-origin ASL web app: static UI and the existing hierarchical SVM API."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from utils import normalize_landmarks

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR.parent / "frontend"


def load_models():
    """Load only the trusted model artifacts shipped with this repository."""
    scaler = joblib.load(MODEL_DIR / "svm_scaler.pkl")
    super_model = joblib.load(MODEL_DIR / "super_model_svm.pkl")
    sub_models = {
        f"G{i}": joblib.load(MODEL_DIR / f"sub_model_svm_G{i}.pkl")
        for i in range(1, 8)
    }
    return scaler, super_model, sub_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.models = load_models()
        logger.info("ASL models loaded")
    except Exception:
        app.state.models = None
        logger.exception("Could not load ASL models")
    yield


app = FastAPI(title="ASL Studio", lifespan=lifespan)


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class HandData(BaseModel):
    landmarks: list[Landmark] = Field(min_length=21, max_length=21)


@app.get("/api/health")
def health():
    if app.state.models is None:
        raise HTTPException(status_code=503, detail="Không tải được mô hình")
    return {"ready": True, "message": "Mô hình sẵn sàng"}


def classify(landmarks: list[Landmark], models):
    scaler, super_model, sub_models = models
    features = normalize_landmarks([landmark.model_dump() for landmark in landmarks])
    scaled = scaler.transform([features])
    group = str(super_model.predict(scaled)[0])
    if group not in sub_models:
        raise ValueError("Unknown prediction group")
    prediction = str(sub_models[group].predict(scaled)[0])
    return {"status": "success", "group": group, "prediction": prediction}


@app.post("/api/predict")
async def predict(data: HandData):
    if app.state.models is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng")
    try:
        return await run_in_threadpool(classify, data.landmarks, app.state.models)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Tọa độ bàn tay không hợp lệ") from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Không thể nhận diện lúc này") from exc


# Keep this mount last so /api routes take precedence.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
