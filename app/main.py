
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib


#  Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (PROJECT_ROOT/ 'models'/ 'complaint_classifier.pkl')


# Create the FastAPI application
app = FastAPI(title='Customer Complaint Routing API',
    description=('NLP-based API for classifying customer complaints '
                 'into financial product categories.'),version='1.0.0')



#Load the trained saved model
model = joblib.load(MODEL_PATH)


#Define request structure
class ComplaintRequest(BaseModel):
    # Complaint text sent by user
    complaint: str

# Define response structure
class PredictionResponse(BaseModel):
    # Predicted  category
    predicted_category: str
    # Highest prediction probability
    confidence: float
    #  status
    status: str

 # health checking
@app.get('/health')
def health_check():
    return {'status': 'healthy','model_loaded': True}

# prediction endpoint
@app.post('/predict',response_model=PredictionResponse)
    # Reject empty input
def predict_complaint(
    request: ComplaintRequest):

    # Remove unnecessary spaces
    complaint = request.complaint.strip()
    # Reject empty input
    if not complaint:
        raise HTTPException(status_code=400,
            detail='Complaint text cannot be empty.')
    try:

        # Predict the most likely category
        prediction = model.predict([complaint])[0]
        # Get probabilities for all categories
        probabilities = model.predict_proba([complaint])[0]
        # Get highest probability
        confidence = float(probabilities.max())

        # Simple confidence-based routing rule
        if confidence < 0.60:
            status = 'Needs Review'
        else:
            status = 'Auto Routed'

        # Return prediction result
        return {
            'predicted_category': prediction,
            'confidence': round( confidence,4),
            'status': status}

    except Exception as error:

        raise HTTPException(status_code=500,
            detail=f'Prediction failed: {str(error)}')