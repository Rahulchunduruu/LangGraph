from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from openai import RateLimitError, AuthenticationError, APIConnectionError, APITimeoutError, APIError
from config import Config
from tools import tools_list
import sqlite3

# Define the tool nodes
tool_nodes=ToolNode(tools_list)



class chatstate(TypedDict):
    question: str
    messages:Annotated[list[BaseMessage], add_messages]

class DocEvalScore(TypedDict):
    score: float
    reason: str

class ChunkScore(TypedDict):
    chunk_index: int
    score: float
    original_chunk:str
    reason: str

def evaluator(state:chatstate):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=Config.OPENAI_API_KEY)
    prompt = """You are an expert evaluator. Evaluate the following query and decide if it requires a tool to answer.
    If the query requires a tool, respond with 'yes'. Otherwise, respond with 'no'.
    question:{0}
    Answer:{1}
    """.format(state['question'],state['messages'])
    llm_eval=llm.with_structured_output(DocEvalScore)
    evaluation = llm_eval.invoke([SystemMessage(content=prompt)])
    return {"messages": [evaluation]}

def extract_rag_chunks(state: chatstate) -> list[str]:
    """Pull chunk texts out of the most recent rag_tool ToolMessage in state."""
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage) and msg.name == 'rag_tool':
            artifact = getattr(msg, 'artifact', None)
            if isinstance(artifact, list) and artifact:
                return [str(c) for c in artifact]
            content = msg.content
            if isinstance(content, list):
                return [getattr(d, 'page_content', str(d)) for d in content]
            if isinstance(content, dict) and 'result' in content:
                return [getattr(d, 'page_content', str(d)) for d in content['result']]
            if isinstance(content, str) and "\n\n---\n\n" in content:
                return [c for c in content.split("\n\n---\n\n") if c.strip()]
            return [str(content)] if content else []
    return []

def grade_chunk(question: str, chunk: str, index: int) -> ChunkScore:
    """Score a single chunk's relevance to the question on a 1-10 scale."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=Config.OPENAI_API_KEY)
    grader = llm.with_structured_output(ChunkScore)
    prompt = f"""You are a strict relevance grader for a retrieval system.
                Rate how directly the CHUNK answers the QUESTION on an integer scale 1-10.


                QUESTION:
                {question}

                CHUNK (index {index}):
                {chunk}

                Return chunk_index={index}, a numeric score, and a one-sentence reason.
                original_chunk in one line summary min 25 words and max 50 worrds"""
    return grader.invoke([SystemMessage(content=prompt)])

RELEVANCE_THRESHOLD = 6.0
MIN_RELEVANT_CHUNKS = 1

def check_each_file(state: chatstate):
    chunks = extract_rag_chunks(state)
    question = state.get('question', '')
    scores: list[ChunkScore] = [
        grade_chunk(question, chunk, i) for i, chunk in enumerate(chunks)
    ]
    #print(f"Question: {question}")
    #for s in scores:
    #    print(f"chunk {s['chunk_index']}: score={s['score']} | {s['reason']} | {s['original_chunk'][:250]}")
    return {"messages": [SystemMessage(content=f"chunk_scores={scores}")]}

def route_after_grading(state: chatstate) -> str:
    """If no chunks were graded or any chunk scored well, answer; otherwise fall back to web search."""
    last = state['messages'][-1]
    content = getattr(last, 'content', '')
    if not isinstance(content, str) or 'chunk_scores=' not in content:
        return 'generate_response'
    if '[]' in content:
        return 'generate_response'
    try:
        scores_str = content.split('chunk_scores=', 1)[1]
        scored = eval(scores_str, {"__builtins__": {}}, {})
        relevant_count = sum(1 for s in scored if float(s['score']) >= RELEVANCE_THRESHOLD)
        if relevant_count >= MIN_RELEVANT_CHUNKS:
            return 'generate_response'
    except Exception:
        return 'generate_response'
    print('content:',content)
    return 'web_fallback'

def web_fallback(state: chatstate):
    """Low-relevance retrieval: search the web and inject the result for the LLM to use."""
    search = DuckDuckGoSearchRun(safesearch='Moderate', max_results=8)
    question = state.get('question', '')
    web_result = search.invoke(question)
    return {"messages": [SystemMessage(content=f"web_search_result:\n{web_result}")]}

def route_after_tools(state: chatstate) -> str:
    """After tools run, only run chunk grading if the rag_tool was used."""
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage):
            return 'check_each_file' if msg.name == 'rag_tool' else 'generate_response'
    return 'generate_response'

def generate_response(state: chatstate):
    #llm = ChatOpenAI(temperature=0.7, openai_api_key=Config.OPENAI_API_KEY)
    prompt="""You are a helpful assistant. Use the tools when needed to answer the user's query and 
           summarize the conversation history and provide a concise response.don't make up any information, if you don't know the answer say you don't know. Always use the tools when needed to get the correct answer."""
    llm = ChatOpenAI( base_url="https://api.x.ai/v1",api_key=Config.XAI_API_KEY,model="grok-4-1-fast-reasoning")
    llm_with_tools = llm.bind_tools(tools_list)
    response = llm_with_tools.invoke([SystemMessage(content=prompt)] + state['messages'])
    
    return {"messages": [response],'question':
                state['messages'][-1].content}

#help to save convestion history in memory, you can customize it to save to a database or file system as needed
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph= StateGraph(chatstate)

graph.add_node('generate_response', generate_response)
graph.add_node('tools', tool_nodes)
graph.add_node('check_each_file', check_each_file)
graph.add_node('web_fallback', web_fallback)

graph.add_edge(START, 'generate_response')
graph.add_conditional_edges('generate_response', tools_condition)
graph.add_conditional_edges(
    'tools',
    route_after_tools,
    {'check_each_file': 'check_each_file', 'generate_response': 'generate_response'},
)
graph.add_conditional_edges(
    'check_each_file',
    route_after_grading,
    {'generate_response': 'generate_response', 'web_fallback': 'web_fallback'},
)
graph.add_edge('web_fallback', 'generate_response')


chat_workflow = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":

    thread_id = "chat_thread_1"
    while True:
        try:
            user_input = input("User Message: ")

            if user_input.strip().lower() in ['exit', 'quit', 'bye']:
                print("Exiting chatbot.")
                break

            initial_state = {
                    'messages': [HumanMessage(content=user_input)]
                        }

            chat_result = chat_workflow.invoke(initial_state,config={"configurable": {"thread_id": thread_id}})

            print("Chatbot Response:", chat_result['messages'][-1].content)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting chatbot.")
            break

        except RateLimitError as e:
            print("API quota exceeded or rate limit hit. You have exceeded your current quota.")
            print("Check your plan, billing details, or wait before retrying.")
            print(f"Details: {e}")

        except AuthenticationError as e:
            print("Authentication failed. Please verify your API key in the config.")
            print(f"Details: {e}")
            break

        except APITimeoutError as e:
            print("The API request timed out. Please try again.")
            print(f"Details: {e}")

        except APIConnectionError as e:
            print("Could not connect to the API. Check your internet connection.")
            print(f"Details: {e}")

        except APIError as e:
            print(f"API returned an error: {e}")
            print("Please try again or type 'exit' to quit.")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("Please try again or type 'exit' to quit.")
