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
2. Start the backend server:
3. Open the testing page:


## Example
If you input:


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

## Disclaimer
This is an educational project only and should not be used for real medical diagnosis.

