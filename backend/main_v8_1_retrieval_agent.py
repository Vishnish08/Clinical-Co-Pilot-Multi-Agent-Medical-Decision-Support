import langchain 

from langchain.chains import RetrievalQA # type: ignore
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

# 1. Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 2. Load vector store
vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=openai_key))

# 3. Initialize the LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=openai_key
)

# 4. Create the Retrieval Agent
retrieval_agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. Query function
def ask(query):
    return retrieval_agent.run(query)

print("Retrieval Agent Ready!")

