import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.predict import get_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Customer Complaint Routing API",
    description="API for classifying customer complaints using NLP",
    version="1.0.0"
)

logger = logging.getLogger(__name__)

# Serve the frontend CSS and JavaScript
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ComplaintRequest(BaseModel):
    complaint: str


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    status: str


# Open the frontend at http://127.0.0.1:8000/
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_complaint(request: ComplaintRequest):
    complaint = request.complaint.strip()

    if not complaint:
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    if len(complaint) > 10000:
        raise HTTPException(status_code=400, detail="Complaint text is too long.")

    try:
        return get_prediction(complaint)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Unable to process the complaint.")
