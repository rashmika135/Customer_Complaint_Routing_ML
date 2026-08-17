# Import required libraries
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
# get the text preprocessing function from preprocessing.py
from src.preprocessing import clean_texts


# project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# path to the cleaned 100,000-row dataset
DATA_PATH = PROJECT_ROOT / "data" / "complaints_working.csv"
# directory where the trained model will be saved
MODEL_DIR = PROJECT_ROOT / "models"
# path where model willbe saved
MODEL_PATH = MODEL_DIR / "complaint_classifier.pkl"



# load the dataset used for model training
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")
print("dataset shape:", df.shape)

# keep only the relevant columns
df = df[["Product","Consumer complaint narrative"]].copy()

# remove rows where the complaints are missing
df = df.dropna(subset=["Consumer complaint narrative"])

# remove rows where the Product label is missing
df = df.dropna( subset=["Product"])
# remove completely duplicated complaint-label pairs
df = df.drop_duplicates(subset=["Consumer complaint narrative","Product"])
df = df.reset_index(drop=True)
print("Cleaned dataset shape:", df.shape)
