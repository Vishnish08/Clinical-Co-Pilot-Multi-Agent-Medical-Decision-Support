from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import re

app = FastAPI()

class Symptoms(BaseModel):
    symptoms: str

# Load model 
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# BASIC SAFETY FILTERS 

def is_empty_or_too_short(text):
    return len(text.strip()) < 3

def contains_abuse_or_hate(text):
    bad_words = ["kill", "suicide", "rape", "fuck", "terror", "attack"]
    return any(word in text.lower() for word in bad_words)

def contains_non_medical_or_junk(text):
    pattern = r"^[0-9@#$%^&*()_+=-]+$"
    return bool(re.match(pattern, text.strip()))

def contains_sexual_content(text):
    sexual_terms = ["sex", "nude", "porn", "boobs", "dick"]
    return any(term in text.lower() for term in sexual_terms)

def looks_like_chat_or_not_symptoms(text):
    conversational_terms = ["hi", "hello", "hey", "what's up", "how are you"]
    return any(term in text.lower() for term in conversational_terms)

# Safety Layer 

def safety_check(text):
    if is_empty_or_too_short(text):
        return False, "Input text is too short. Provide valid symptoms."

    if contains_abuse_or_hate(text):
        return False, "Unsafe or harmful content detected. Cannot process."

    if contains_sexual_content(text):
        return False, "Sexual content detected. Not allowed."

    if contains_non_medical_or_junk(text):
        return False, "Input appears invalid (nonsensical characters)."

    if looks_like_chat_or_not_symptoms(text):
        return False, "Provide symptoms, not a greeting or chat message."

    return True, "Safe"

# API ENDPOINT 

@app.post("/predict")
async def predict(symptoms: Symptoms):

    #  RUN SAFETY LAYER 
    safe, message = safety_check(symptoms.symptoms)

    if not safe:
        return {
            "safe": False,
            "message": message,
            "prediction": None,
            "probabilities": None
        }

    # If safe, process normally
    X_input = vectorizer.transform([symptoms.symptoms])
    proba = model.predict_proba(X_input)[0]
    pred_class = model.classes_[np.argmax(proba)]

    return {
        "safe": True,
        "prediction": pred_class,
        "probabilities": dict(zip(model.classes_, proba.tolist()))
    }

