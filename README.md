# Clinical-Co-Pilot-Multi-Agent-Medical-Decision-Support
A multi-agent medical decision support system with symptom analysis, diagnosis suggestions, and clinical summaries.



Clinical Co-Pilot is a simple medical decision-support tool that takes symptoms as input and provides a possible diagnosis, suggested tests, and a short clinical summary. This project is meant for learning, portfolio building, and demonstrating how a basic healthcare AI assistant can be structured.

## What the project does
- Takes symptoms from the user in plain text
- Applies basic rule-based logic to interpret the symptoms
- Returns:
  - A possible medical condition
  - Recommended diagnostic tests
  - A short summary that looks like a doctor’s note

## How the backend works
The backend is built using FastAPI.  
You run it locally, and it exposes an API endpoint where you can send symptom text and receive the model’s output.  
You can also test the API easily using the built-in Swagger UI.

## How to run the backend
1. Install the required packages:
   pip install fastapi uvicorn
   
2. Start the backend server:
   uvicorn main:app --reload

3. Open the testing page:
   Go to: http://127.0.0.1:8000/docs

## Example
If you input: I have fever, headache, and sore throat for two days.


The system may respond with:
- Likely viral infection  
- Suggested tests like CBC and CRP  
- A short clinical summary of your symptoms and assessment  

## Purpose of this project
This project is a starting point for building a larger medical AI system.  
Future versions can include:
- Better symptom extraction  
- Machine learning models  
- A proper UI  
- Advanced reasoning agents

## How the project works (Architecture)
The project works in a simple layered structure:

- **Input Layer:** The user provides symptoms in plain text.  
- **Intelligence Layer:** Small agents analyze the symptoms, suggest possible conditions, and recommend tests.  
- **Output Layer:** Provides a clean summary including diagnosis, suggested tests, and short clinical notes.  
- **Optional Logging Layer:** Stores inputs and outputs for later improvements.


## Disclaimer
This is an educational project only and should not be used for real medical diagnosis.

