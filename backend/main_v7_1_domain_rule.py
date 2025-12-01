from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import re

app = FastAPI()

class Symptoms(BaseModel):
    symptoms: str

# Load pre-trained models 
xgb_model = joblib.load("models/best_xgb.pkl")
lgb_model = joblib.load("models/best_lgb.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

weight_xgb = 0.6
weight_lgb = 0.4


# 🔒 VERSION 7.1 — DOMAIN RULE LAYER
NON_MEDICAL_KEYWORDS = [
    # Finance
    "stock", "crypto", "bitcoin", "loan", "salary", "business", "investment",

    # Education / Career
    "exam", "job", "internship", "college", "engineering",

    # Cooking / Food
    "recipe", "cook", "kitchen",

    # Technology
    "laptop", "mobile", "computer", "coding",

    # Astrology
    "horoscope", "zodiac", "kundli",

    # Sports
    "football", "cricket", "basketball",

    # General non-medical
    "weather", "travel", "flight", "train"
]

def violates_domain_rules(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in NON_MEDICAL_KEYWORDS)


# Helper for ensemble voting
def weighted_prob_vote(proba_xgb, proba_lgb, classes):
    weighted_proba = weight_xgb * proba_xgb + weight_lgb * proba_lgb
    final_index = np.argmax(weighted_proba)
    return classes[final_index], weighted_proba.tolist()

# API ENDPOINT
@app.post("/predict")
async def predict(symptoms: Symptoms):
    user_input = symptoms.symptoms.strip()

    # 1️⃣ DOMAIN RULE CHECK
    if violates_domain_rules(user_input):
        return {
            "error": "❌ This system only handles medical symptom descriptions. "
                     "Please describe health-related symptoms only."
        }

    # 2️⃣ Transform text
    X_input = vectorizer.transform([user_input])

    # 3️⃣ Model probabilities
    proba_xgb = xgb_model.predict_proba(X_input)[0]
    proba_lgb = lgb_model.predict_proba(X_input)[0]
    classes = xgb_model.classes_

    # 4️⃣ Ensemble output
    final_label, final_proba = weighted_prob_vote(proba_xgb, proba_lgb, classes)

    return {
        "prediction": final_label,
        "probabilities": dict(zip(classes, final_proba))
    }
