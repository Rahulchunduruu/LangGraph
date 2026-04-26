import os
import streamlit as st
from main import chat_workflow
from langchain_core.messages import AIMessage, HumanMessage
from openai import RateLimitError, AuthenticationError, APIConnectionError, APITimeoutError, APIError
from tools import add_to_vectorstore

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
VECTORSTORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vectorstores",
    "vector_data_faiss",
)

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)


st.title("Chatbot with LangGraph")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for message in st.session_state['chat_history']:
    with st.chat_message(message['role'], avatar=message['avatar']):
        st.write(message['content'])


user_input = st.chat_input("Type your message here...", accept_file=True)

if user_input:
    user_text = user_input.text
    user_files = user_input.files

    if user_files:
        uploaded_file = user_files[0]
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        saved_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.read())
        try:
            add_to_vectorstore(file_path=saved_path, vectorstore_path=VECTORSTORE_PATH)
            st.success(f"Saved '{uploaded_file.name}' to documents and indexed it.")
        except Exception as e:
            st.error(f"Failed to process the uploaded file: {e}")

    if user_text:
        st.session_state['chat_history'].append(
            {"role": "user", "content": user_text, "avatar": "https://picsum.photos/id/237/200/300"}
        )
        with st.chat_message("user", avatar="https://picsum.photos/id/237/200/300"):
            st.write(user_text)

        with st.chat_message("assistant", avatar="https://picsum.photos/100"):
            def token_stream():
                for chunk, metadata in chat_workflow.stream(
                    {'messages': [HumanMessage(content=user_text)],'question':user_text},
                    config={"configurable": {"thread_id": 'chat_thread_1'}},
                    stream_mode="messages",
                ):
                    if metadata.get('langgraph_node') != 'generate_response':
                        continue
                    if isinstance(chunk, AIMessage) and chunk.content and not chunk.tool_call_chunks:
                        yield chunk.content

            full_response = None
            try:
                full_response = st.write_stream(token_stream())
            except RateLimitError as e:
                full_response = "API quota exceeded. You have exceeded your current quota. Please check your plan and billing details."
                st.error(full_response)
                st.caption(f"Details: {e}")
            except AuthenticationError as e:
                full_response = "Authentication failed. Please verify your API key in the config."
                st.error(full_response)
                st.caption(f"Details: {e}")
            except APITimeoutError as e:
                full_response = "The API request timed out. Please try again."
                st.warning(full_response)
                st.caption(f"Details: {e}")
            except APIConnectionError as e:
                full_response = "Could not connect to the API. Check your internet connection."
                st.error(full_response)
                st.caption(f"Details: {e}")
            except APIError as e:
                full_response = f"API returned an error: {e}"
                st.error(full_response)
            except Exception as e:
                full_response = f"An unexpected error occurred: {e}"
                st.error(full_response)

        st.session_state['chat_history'].append(
            {"role": "assistant", "content": full_response, "avatar": "https://picsum.photos/100"}
        )
