
from pathlib import Path
import joblib


# Load the trained model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "complaint_classifier.pkl"

model = joblib.load(MODEL_PATH)


def get_prediction(complaint):

    # Predict the complaint category
    category = str(model.predict([complaint])[0])

    # Get the highest prediction probability
    probabilities = model.predict_proba([complaint])[0]
    confidence = float(probabilities.max())

    # Decide whether the complaint needs manual review
    if confidence >= 0.60:
        status = "Auto Routed"
    else:
        status = "Needs Review"

    return {
        "predicted_category": category,
        "confidence": round(confidence, 4),
        "status": status
    }
