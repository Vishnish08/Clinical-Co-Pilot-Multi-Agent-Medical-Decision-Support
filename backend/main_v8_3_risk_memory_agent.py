from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain.chains import RetrievalQA  # type: ignore
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


# Load Vectorstore (FAISS)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()



# The LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_KEY
)



# PROMPT: Risk + Memory Reasoning
risk_prompt = """
You are a clinical risk assessment agent.

### Responsibilities:
1. Extract measurable symptoms from user input.
2. Pull correct medical facts from the retrieval system.
3. Assess patient risk level as ONLY one of: Low, Moderate, High.
4. Provide clear reasoning.
5. Do NOT provide treatment unless asked.
6. If information is missing, say "Insufficient clinical data."
7. Never hallucinate medical conditions.

### Output Format:
- Extracted Symptoms
- Retrieved Medical Facts
- Clinical Reasoning
- Risk Category (Low / Moderate / High)
- Important Safety Warning
"""


# Create Retrieval + Reasoning Agent
risk_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": risk_prompt}
)

# Query Function
def assess_risk(query: str):
    """Run the risk memory agent."""
    return risk_agent.run(query)


# Debug
if __name__ == "__main__":
    print("Risk Memory Agent v8.3 Ready!")
    example = assess_risk("Patient has chest pain and sweating. What is risk level?")
    print(example)
