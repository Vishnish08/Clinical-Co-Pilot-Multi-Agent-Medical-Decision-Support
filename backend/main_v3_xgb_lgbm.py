from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = FastAPI()

class Symptoms(BaseModel):
    symptoms: str

# Sample training data
symptoms_list = [
    "fever cough fatigue",
    "headache nausea dizziness",
    "chest pain shortness of breath",
    "vomiting stomach pain diarrhea",
    "joint pain swelling stiffness"
]

labels = [
    "flu",
    "migraine",
    "heart issue",
    "food poisoning",
    "arthritis"
]

# Convert string labels → numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(symptoms_list)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X, y)

# Train LightGBM model
lgbm_model = LGBMClassifier()
lgbm_model.fit(X, y)

@app.post("/predict")
async def predict(symptoms: Symptoms):
    X_input = vectorizer.transform([symptoms.symptoms])

    # XGBoost prediction
    xgb_pred_num = xgb_model.predict(X_input)[0]
    xgb_pred_label = label_encoder.inverse_transform([xgb_pred_num])[0]

    # LightGBM prediction
    lgbm_pred_num = lgbm_model.predict(X_input)[0]
    lgbm_pred_label = label_encoder.inverse_transform([lgbm_pred_num])[0]

    return {
        "xgboost_prediction": xgb_pred_label,
        "lightgbm_prediction": lgbm_pred_label
    }

