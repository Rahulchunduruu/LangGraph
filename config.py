import os
from dotenv import load_dotenv

load_dotenv()

class Config:   
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    XAI_API_KEY = os.getenv('XAI_API_KEY')
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')