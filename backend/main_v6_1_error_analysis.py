from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

app = FastAPI()

class EvalItem(BaseModel):
    symptoms: str
    true_label: str

# Load models 
xgb_model = joblib.load("models/best_xgb.pkl")
lgb_model = joblib.load("models/best_lgb.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Store all evaluation samples 
EVAL_SYMPTOMS = []
TRUE_LABELS = []
PRED_XGB = []
PRED_LGB = []

@app.post("/evaluate")
async def evaluate(item: EvalItem):

    # Transform symptoms
    X = vectorizer.transform([item.symptoms])

    # Predictions
    px = xgb_model.predict(X)[0]
    pl = lgb_model.predict(X)[0]

    # Save for analysis
    EVAL_SYMPTOMS.append(item.symptoms)
    TRUE_LABELS.append(item.true_label)
    PRED_XGB.append(px)
    PRED_LGB.append(pl)

    return {
        "true_label": item.true_label,
        "xgb_prediction": px,
        "lgb_prediction": pl
    }


@app.get("/report")
async def error_report():

    # If not enough samples
    if len(TRUE_LABELS) < 5:
        return {"error": "Not enough evaluation data. Add at least 5 samples."}

    # Confusion matrices
    cm_xgb = confusion_matrix(TRUE_LABELS, PRED_XGB).tolist()
    cm_lgb = confusion_matrix(TRUE_LABELS, PRED_LGB).tolist()

    # Reports
    report_xgb = classification_report(TRUE_LABELS, PRED_XGB, output_dict=True)
    report_lgb = classification_report(TRUE_LABELS, PRED_LGB, output_dict=True)

    # Collect mistakes
    mistakes = []

    for i in range(len(TRUE_LABELS)):
        if TRUE_LABELS[i] != PRED_XGB[i] or TRUE_LABELS[i] != PRED_LGB[i]:
            mistakes.append({
                "symptoms": EVAL_SYMPTOMS[i],
                "true_label": TRUE_LABELS[i],
                "xgb_pred": PRED_XGB[i],
                "lgb_pred": PRED_LGB[i]
            })

    return {
        "total_samples": len(TRUE_LABELS),
        "confusion_matrix_xgb": cm_xgb,
        "confusion_matrix_lgb": cm_lgb,
        "classification_report_xgb": report_xgb,
        "classification_report_lgb": report_lgb,
        "mistakes": mistakes
    }

