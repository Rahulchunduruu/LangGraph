import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

class Config:    
    #If you prefer to run this in the terminal, make sure the first line is active and the second is commented out
    #OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    #XAI_API_KEY = os.getenv('XAI_API_KEY')
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    XAI_API_KEY = st.secrets["XAI_API_KEY"]
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please add your API key.")
    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY not found in .env file. Please add your API key.")
