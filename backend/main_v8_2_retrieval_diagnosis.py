from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
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


# The LLM with Safety Settings
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_KEY
)


# Safety Prompt
diagnosis_prompt = """
You are a clinical reasoning assistant.
Your responsibilities:

1. Use ONLY the retrieved medical facts from the vector index.
2. If information is missing, say “Not enough data”.
3. Provide possible differential diagnosis but NEVER claim certainty.
4. Avoid giving treatment unless explicitly asked.
5. NEVER invent patient data.
6. Maintain a professional, medically safe tone.

Final Output Format:
- Key Retrieved Facts
- Logical Reasoning
- Differential Diagnosis (3–5 possibilities)
- Warning if insufficient context
"""


# Create Retrieval Agent
diagnosis_agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": diagnosis_prompt}
)


# Query Function
def diagnose(query: str):
    """Run the diagnosis agent."""
    return diagnosis_agent.run(query)

# Debug
if __name__ == "__main__":
    print("Diagnosis Agent v8.2 Ready!")
    test = diagnose("Patient has high fever and rash. Possible reason?")
    print(test)
