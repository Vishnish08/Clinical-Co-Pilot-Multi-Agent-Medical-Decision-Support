from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load pre-trained models & vectorizer
vectorizer = joblib.load("vectorizer.pkl")      
best_xgb = joblib.load("best_xgb.pkl")          
best_lgb = joblib.load("best_lgb.pkl")           

# Define class labels (same as training)
class_labels = ['arthritis', 'flu', 'food poisoning', 'heart issue', 'migraine']


# FastAPI setup
app = FastAPI()

class Symptoms(BaseModel):
    symptoms: str

# Prediction endpoint
@app.post("/predict")
async def predict(symptoms: Symptoms):
    # Transform input text
    X_input = vectorizer.transform([symptoms.symptoms])

    # Get predicted probabilities
    xgb_probs = best_xgb.predict_proba(X_input)
    lgb_probs = best_lgb.predict_proba(X_input)

  
    # Weighted ensemble (you can change weights)
    weight_xgb = 0.6
    weight_lgb = 0.4

    combined_probs = weight_xgb * xgb_probs + weight_lgb * lgb_probs

    # Final predicted label
    pred_index = np.argmax(combined_probs, axis=1)[0]
    prediction = class_labels[pred_index]

    return {"prediction": prediction, "probabilities": combined_probs.tolist()}

