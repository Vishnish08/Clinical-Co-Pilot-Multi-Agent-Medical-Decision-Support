from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

app = FastAPI()

class Symptoms(BaseModel):
    symptoms: str

# --- Load models and vectorizer if they exist ---
if os.path.exists("models/best_xgb.pkl") and os.path.exists("models/best_lgb.pkl") and os.path.exists("models/vectorizer.pkl"):
    print("Loading saved models...")
    best_xgb = joblib.load("models/best_xgb.pkl")
    best_lgb = joblib.load("models/best_lgb.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
else:
    # If models don't exist, train and save (first-time setup)
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV

    print("No saved models found. Training models...")

    # Sample training data
    symptoms_list = [
        "fever cough fatigue",
        "headache nausea dizziness",
        "chest pain shortness of breath",
        "joint pain stiffness arthritis",
        "vomiting diarrhea food poisoning"
    ]
    labels = ["flu", "migraine", "heart issue", "arthritis", "food poisoning"]

    # Feature extraction
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(symptoms_list)

    # --- XGBoost tuning ---
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    param_grid_xgb = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200]
    }
    grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='accuracy')
    grid_xgb.fit(X, labels)
    best_xgb = grid_xgb.best_estimator_
    print("Best XGB params:", grid_xgb.best_params_)

    # --- LightGBM tuning ---
    lgb_model = LGBMClassifier()
    param_grid_lgb = {
        'num_leaves': [7, 15, 31],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200]
    }
    grid_lgb = GridSearchCV(lgb_model, param_grid_lgb, cv=3, scoring='accuracy')
    grid_lgb.fit(X, labels)
    best_lgb = grid_lgb.best_estimator_
    print("Best LGBM params:", grid_lgb.best_params_)

    # Save models and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_xgb, "models/best_xgb.pkl")
    joblib.dump(best_lgb, "models/best_lgb.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("Models saved. Server ready!")

# --- Prediction endpoint ---
@app.post("/predict")
async def predict(symptoms: Symptoms):
    X_input = vectorizer.transform([symptoms.symptoms])
    pred_xgb = best_xgb.predict(X_input)[0]
    pred_lgb = best_lgb.predict(X_input)[0]

    # Return both predictions (for V5 ensemble later)
    return {"prediction_xgb": pred_xgb, "prediction_lgb": pred_lgb}

