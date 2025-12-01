# **Clinical Co-Pilot – Multi-Agent Medical Decision Support System**

#### **Clinical Co-Pilot** is a lightweight, multi-agent medical decision-support system that accepts symptoms as input and generates a likely diagnosis, recommended tests, and a short clinical summary.

#### The project is designed for **learning**, **experimentation**, and **portfolio showcase**, demonstrating how a simple medical AI backend can be built using **FastAPI**, **LLMs**, and **light reasoning agents**.

---

## **✨ Features**

1. **Plain-text symptom input**
2. **Reasoning + rules + LLM processing**
3. **Returns:**
   - Likely medical condition  
   - Recommended diagnostic tests  
   - Doctor-style clinical summary  
4. **PDF report generation supported**
5. **FastAPI backend with Swagger UI**
6. **Fully local backend, easy to run**

---

## **🧠 System Overview**

### **1. Input Layer**
Users enter free-form symptoms such as:  
“Fever, sore throat, fatigue for 2 days.”

### **2. Intelligence Layer**
The backend combines:

- Lightweight rule-based logic  
- LLM reasoning  
- Simple confidence computations  

### **3. Output Layer**
The system returns:

- **Likely Diagnosis**  
- **Suggested Diagnostic Tests**  
- **Clinical Summary Note**  
- **Optional PDF Report**

---

## **📂 Project Structure**
├── main.py # Backend (FastAPI + LLM + PDF + logic)
├── requirements.txt # Dependencies
├── README.md # Documentation
└── .env (created by user) # API key storage


#### **There are no models, no utils, no extra backend folders.**  
#### **Everything is contained inside main.py for clarity.**

---

## **🛠 Installation & Setup**

### **1. Clone the repository**
git clone https://github.com/YOUR_USERNAME/Clinical-Co-Pilot-Multi-Agent-Medical-Decision-Support

### **2. Install dependencies**
pip install -r requirements.txt

### **3. Add your environment variables**
Create a `.env` file:
OPENAI_API_KEY=your_api_key_here

### **4. Run the backend**
uvicorn main:app --reload

### **5. Open Swagger UI**
http://127.0.0.1:8000/docs


From here, you can test the API without writing any code.

---

## **📘 Example**

### **Input**
"I have fever, headache, and throat pain."

### **Output**
1. **Condition:** Possible Viral Infection  
2. **Recommended Tests:** CBC, CRP  
3. **Clinical Summary:** A structured note summarizing symptoms and assessment  

---

## **🎯 Purpose of This Project**

This project serves as:

1. A **beginner-friendly medical AI backend**
2. A **portfolio showcase** for ML + AI + FastAPI
3. A foundation for future enhancements, including:
   - Machine learning models  
   - Medical knowledge graphs  
   - Multi-agent architectures  
   - Structured UI dashboard  
   - Electronic medical history retrieval  

---

## **⚠️ Disclaimer**

#### This system is for **educational, experimental, and portfolio purposes only.**  
#### It must **NOT** be used for real medical diagnosis, treatment, or patient care.

---


