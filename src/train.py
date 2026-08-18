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


# input and target columns
# X = the raw customer complaint text
X = df["Consumer complaint narrative"]
# y = the complaint category that the model should predict
y = df["Product"]

# create the train / validation / test split
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.30,stratify=y,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.50,stratify=y_temp,random_state=42)

# combine training and validation data
X_final_train = pd.concat([X_train, X_val],ignore_index=True)
y_final_train = pd.concat([y_train, y_val],ignore_index=True)
print("\nFinal training samples:", len(X_final_train))
print("test samples:", len(X_test))

# Create the final NLP + ML pipeline
# 1. Cleans the text
# 2. Converts text into TF-IDF features
# 3. Predicts the complaint category using Logistic Regression
production_pipeline = Pipeline([("cleaner",FunctionTransformer(clean_texts,validate=False)),
# convert cleaned text into TF-IDF features
 ("tfidf",TfidfVectorizer( max_features=30000,ngram_range=(1, 2))),
 # final model selected during jupyter notebook
 ("model",LogisticRegression(C=1.0, max_iter=1000))])


#train the final production model
print("\nTraining final production model...")
# train the complete pipeline using raw complaint text
production_pipeline.fit( X_final_train,y_final_train)
print("Training completed.")


#create the models directory 
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Save the cleaner + TF-IDF vectorizer + Logistic Regression model

joblib.dump(production_pipeline,MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)

# test the saved pipeline with one sample complaint
# Create a sample raw customer complaint
sample_complaint = ["My card was charged twice and I do not recognize one of the transactions."]

# make a prediction 
sample_prediction = production_pipeline.predict(sample_complaint)
print("\nSample complaint:")
print(sample_complaint[0])

print("\nPredicted category:")
print(sample_prediction[0])
# create a sample complaint for checking the models predictionand confidence scores
sample_complaint = ["My card was charged twice and I do not recognize one of the transactions."]

# predict the most likely complaint category
prediction = production_pipeline.predict(sample_complaint)
# get the probability for each possible category
probabilities = production_pipeline.predict_proba(sample_complaint)[0]
# get category names in the same order as the probabilities
classes = production_pipeline.named_steps["model"].classes_
# combine category names with their probabilities
results = list(zip(classes, probabilities))
# sort categories from highest probability to lowest probability
results = sorted(results,key=lambda x: x[1],reverse=True)
# Display the predicted category
print("\nSample complaint:")
print(sample_complaint[0])
print("\nPredicted category:")
print(prediction[0])
# Display confidence scores for all categories
print("\nCategory probabilities:")
for category, probability in results:
    print(f"{category}: {probability:.4f}")