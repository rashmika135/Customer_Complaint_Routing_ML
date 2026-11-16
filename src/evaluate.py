
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,f1_score,classification_report,confusion_matrix)


# File paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "complaints_working.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "complaint_classifier.pkl"


# Load the dataset
df = pd.read_csv(DATA_PATH)

df = df[["Product", "Consumer complaint narrative"]].copy()
df = df.dropna(subset=["Product", "Consumer complaint narrative"])
df = df.drop_duplicates(subset=["Consumer complaint narrative", "Product"])
df = df.reset_index(drop=True)

X = df["Consumer complaint narrative"]
y = df["Product"]


# Recreate the same data split used in train.py
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    stratify=y_temp, random_state=42)


# Load the saved model
model = joblib.load(MODEL_PATH)

# Predict the test complaints
predictions = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

print("Final Model Evaluation")
print("----------------------")
print(f"Test samples: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
