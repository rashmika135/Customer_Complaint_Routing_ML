
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
