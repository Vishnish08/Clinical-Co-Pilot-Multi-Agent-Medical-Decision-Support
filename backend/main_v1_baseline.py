from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Clinical Co-Pilot V1")

class SymptomInput(BaseModel):
    symptoms: str

@app.post("/predict")
def predict_condition(data: SymptomInput):
    symptoms = data.symptoms.lower()

    # Simple rule-based logic
    if "fever" in symptoms and "cough" in symptoms:
        diagnosis = "Likely Viral Infection"
        tests = ["CBC", "CRP"]
    elif "chest pain" in symptoms:
        diagnosis = "Possible Cardiac Issue"
        tests = ["ECG", "Troponin Test"]
    else:
        diagnosis = "Not enough information"
        tests = []

    summary = f"Patient reports: {data.symptoms}. Assessment: {diagnosis}."

    return {
        "diagnosis": diagnosis,
        "suggested_tests": tests,
        "summary": summary
    }
