from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, HumanInputRun
from tavily import TavilyClient
from config import Config

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,TextLoader,WebBaseLoader
from langchain_openai import OpenAIEmbeddings

import os
import yfinance as yf

#loading documents based on file type, you can customize this function to support more file types or data sources as needed
def loading_documents(file_path: str) -> list:

    if file_path.endswith('.txt'):
        Text_loader = TextLoader(file_path)
        return Text_loader.load()
    elif file_path.endswith('.pdf'):
        PyPDF_loader = PyPDFLoader(file_path)
        return PyPDF_loader.load()
    elif file_path.startswith('http'): 
        WebBase_loader = WebBaseLoader(file_path)
        return WebBase_loader.load()
    else:
        raise ValueError("Unsupported file format. Only .txt and .pdf are supported.")
    
 #call this function to load documents and split into chunks
def load_and_split_documents(file_path: str) -> list:
    documents = loading_documents(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return text_splitter.split_documents(documents)   
        
def create_vectorstore(file_path: str) -> FAISS:
    docs = load_and_split_documents(file_path=file_path)
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def save_vectorstore(vectorstore: FAISS, file_path: str):
    vectorstore.save_local(file_path)

def add_to_vectorstore(file_path: str, vectorstore_path: str) -> FAISS:
    """Append a new document's chunks to an existing FAISS index, or create one if missing."""
    docs = load_and_split_documents(file_path=file_path)
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)

    if os.path.isdir(vectorstore_path) and os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(vectorstore_path)
    return vectorstore

def retrieve_from_vectorstore(vectorstore_path: str, query: str) -> list:
    if not os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
        return []
    vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY), allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=5)
    return results


#TOOLS
Search=DuckDuckGoSearchRun(safesearch="Moderate", max_results=12,query="provide a concise answer to the question based on the search results, and include relevant sources links")



@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
        #return {"message": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(stockname: str) -> dict:
    """Fetch current stock price using yfinance."""
    try:
        ticker = yf.Ticker(stockname.upper())
        data = ticker.history(period="1d")
        
        if data.empty:
            return {"error": f"No data found for {stockname}"}
        
        latest = data.iloc[-1]
        return {
            "ticker": stockname.upper(),
            "price": float(latest['Close']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "volume": int(latest['Volume'])
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def decision_maker(question: str) -> str:
    """
    Make a simple yes/no decision based on the input text.
    """
    if "no" in question.lower():
        return "no"
    return "yes"

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    HIL=HumanInputRun()
    decision = HIL.run(f"Approve buying {quantity} shares of {symbol}? (yes/no)")


    if decision_maker(decision) == "yes":
            return {
                "status": "success",
                "message": f"Purchase order placed for {quantity} shares of {symbol}.",
                "symbol": symbol,
                "quantity": quantity,
            }
        
    else:
            return {
                "status": "cancelled",
                "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
                "symbol": symbol,
                "quantity": quantity,
            }

_VECTORSTORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'vectorstores',
    'vector_data_faiss',
)
_DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'documents')

os.makedirs(_DOCUMENTS_DIR, exist_ok=True)
os.makedirs(_VECTORSTORE_PATH, exist_ok=True)



@tool(response_format="content_and_artifact")
def rag_tool(query: str) -> tuple[str, list[str]]:
    """
    Perform Retrieval-Augmented Generation (RAG) to fetch relevant information based on the query.
    """
    docs = retrieve_from_vectorstore(_VECTORSTORE_PATH, query)
    chunks = [getattr(d, 'page_content', str(d)) for d in docs]
    content = "\n\n---\n\n".join(chunks) if chunks else "No relevant chunks found."
    return content, chunks


tools_list=[Search,calculator,get_stock_price,rag_tool,purchase_stock]
