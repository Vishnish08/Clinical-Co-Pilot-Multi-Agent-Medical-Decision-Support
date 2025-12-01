from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class Symptoms(BaseModel):
    symptoms: str

# Load models 
xgb_model = joblib.load("models/best_xgb.pkl")
lgb_model = joblib.load("models/best_lgb.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

weight_xgb = 0.6
weight_lgb = 0.4


# ⭐ VERSION 7.2 – Mini Knowledge Graph
KNOWLEDGE_GRAPH = {
    "fever": {
        "possible_conditions": ["Viral Fever", "Flu", "Dengue", "Typhoid"],
        "severity": "medium",
        "advice": "Monitor temperature, stay hydrated, seek help if fever > 102°F for 3 days."
    },
    "chest pain": {
        "possible_conditions": ["Heart Attack", "Angina", "Muscle Strain"],
        "severity": "high",
        "advice": "⚠️ Chest pain with sweating or radiating pain → emergency medical help required."
    },
    "headache": {
        "possible_conditions": ["Migraine", "Tension Headache", "Sinusitis"],
        "severity": "low",
        "advice": "Rest, hydration; if persistent or with vision issues, consult doctor."
    },
    "cough": {
        "possible_conditions": ["Bronchitis", "Asthma", "Common Cold"],
        "severity": "low",
        "advice": "Warm fluids, avoid cold air; if >3 weeks, get chest check-up."
    },
    "breathlessness": {
        "possible_conditions": ["Asthma Attack", "COPD", "Heart Failure"],
        "severity": "high",
        "advice": "⚠️ Sudden breathlessness = emergency help required immediately."
    }
}

def get_knowledge_graph_info(user_text: str):
    user_text = user_text.lower()
    for symptom in KNOWLEDGE_GRAPH.keys():
        if symptom in user_text:
            return KNOWLEDGE_GRAPH[symptom]
    return None  # No matching knowledge node



# Helper: Weighted Ensemble
def weighted_prob_vote(proba_xgb, proba_lgb, classes):
    weighted_proba = weight_xgb * proba_xgb + weight_lgb * proba_lgb
    final_index = np.argmax(weighted_proba)
    return classes[final_index], weighted_proba.tolist()


# API ENDPOINT
@app.post("/predict")
async def predict(symptoms: Symptoms):
    user_input = symptoms.symptoms.strip()

    # 1️⃣ Get Knowledge Graph Info
    kg_info = get_knowledge_graph_info(user_input)

    # 2️⃣ Convert text → vector
    X_input = vectorizer.transform([user_input])

    # 3️⃣ Model probabilities
    proba_xgb = xgb_model.predict_proba(X_input)[0]
    proba_lgb = lgb_model.predict_proba(X_input)[0]
    classes = xgb_model.classes_

    # 4️⃣ Ensemble
    final_label, final_proba = weighted_prob_vote(proba_xgb, proba_lgb, classes)

    return {
        "model_prediction": final_label,
        "probabilities": dict(zip(classes, final_proba)),
        "knowledge_graph": kg_info if kg_info else "No KG match found"
    }

