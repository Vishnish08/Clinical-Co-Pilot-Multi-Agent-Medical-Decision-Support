from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI()

class Symptoms(BaseModel):
    symptoms: str

# ----- Load pre-trained & tuned models -----
xgb_model = joblib.load("models/best_xgb.pkl")
lgb_model = joblib.load("models/best_lgb.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ----- Ensemble weights -----
weight_xgb = 0.6
weight_lgb = 0.4

# ----- Helper function: weighted probability vote -----
def weighted_prob_vote(proba_xgb, proba_lgb, classes):
    # Weighted sum of probabilities
    weighted_proba = weight_xgb * proba_xgb + weight_lgb * proba_lgb
    # Get final class
    final_index = np.argmax(weighted_proba)
    return classes[final_index], weighted_proba.tolist()

# ----- FastAPI endpoint -----
@app.post("/predict")
async def predict(symptoms: Symptoms):
    X_input = vectorizer.transform([symptoms.symptoms])
    
    # Predict probabilities
    proba_xgb = xgb_model.predict_proba(X_input)[0]
    proba_lgb = lgb_model.predict_proba(X_input)[0]
    
    classes = xgb_model.classes_  # Assuming both models have same classes
    
    # Ensemble with weighted probabilities
    final_label, final_proba = weighted_prob_vote(proba_xgb, proba_lgb, classes)
    
    return {
        "prediction": final_label,
        "probabilities": dict(zip(classes, final_proba))
    }

