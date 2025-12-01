from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()  


from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

app = FastAPI()

# Define input schema
class Symptoms(BaseModel):
    symptoms: str

@app.post("/predict")
async def predict(symptoms: Symptoms):
    return {"prediction": "test"}

# Sample training data (replace with your real dataset)
symptoms_list = [
    "fever cough fatigue",
    "headache nausea dizziness",
    "chest pain shortness of breath"
]
labels = ["flu", "migraine", "heart issue"]

# Feature engineering + model pipeline
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(symptoms_list)
model = LogisticRegression()
model.fit(X, labels)

# Save vectorizer and model for reuse (optional)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

@app.post("/predict")
async def predict(symptoms: Symptoms):
    # Transform input text into features
    X_input = vectorizer.transform([symptoms.symptoms])
    prediction = model.predict(X_input)
    return {"prediction": prediction[0]}
