# Chatbot CRAG (Corrective RAG with LangGraph)

A LangGraph-powered chatbot that combines **tool use**, **Retrieval-Augmented Generation (RAG)** over a FAISS vector store, **per-chunk relevance grading**, and a **web-search fallback** when the local knowledge base is not relevant enough. The UI is built with Streamlit and supports document upload (PDF/TXT) for on-the-fly indexing.

## Features

- LangGraph state machine with conditional routing
- Tools: calculator, stock-price lookup (Tavily), RAG over FAISS, human-in-the-loop stock purchase
- **Corrective RAG**: each retrieved chunk is graded 1-10; if no chunk meets the threshold, the graph falls back to a DuckDuckGo web search
- SQLite-backed conversation memory via `SqliteSaver`
- Streamlit chat UI with file upload that incrementally indexes documents into FAISS
- Streaming responses filtered to surface only final-answer tokens (grader JSON is hidden)

## Project Structure

```
chatbot_CRAG/
├── app.py                 # Streamlit UI + token streaming
├── main.py                # LangGraph definition (nodes, edges, compile)
├── tools.py               # Tools + RAG / FAISS helpers
├── config.py              # Env-based API key loading
├── requirements.txt
├── chatbot.db             # SQLite checkpointer (auto-created)
├── documents/             # Uploaded source files (auto-created)
└── vectorstores/
    └── vector_data_faiss/ # FAISS index (auto-created)
```

## Setup

1. Create a virtual environment and install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file alongside `config.py`:
   ```env
   OPENAI_API_KEY=sk-...
   XAI_API_KEY=xai-...
   TAVILY_API_KEY=tvly-...
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
   Or run the CLI loop:
   ```bash
   python main.py
   ```

## Nodes Architecture

The graph is defined in [main.py](main.py) and compiled with a SQLite checkpointer.

### Flow Diagram

```
                              ┌───────────┐
                              │   START   │
                              └─────┬─────┘
                                    ▼
                         ┌────────────────────┐
                         │ generate_response  │  ◄────────────┐
                         │ (LLM + bind_tools) │               │
                         └──────────┬─────────┘               │
                                    │                         │
                       tools_condition (auto)                 │
                ┌───────────────────┴──────────────┐          │
                ▼                                  ▼          │
          ┌──────────┐                          ┌─────┐       │
          │   END    │                          │tools│       │
          └──────────┘                          └──┬──┘       │
                                                   │          │
                                       route_after_tools      │
                              ┌────────────────────┴───┐      │
                              ▼                        ▼      │
                   ┌──────────────────┐         (any other    │
                   │ check_each_file  │          tool)        │
                   │ (grade chunks)   │            │          │
                   └────────┬─────────┘            │          │
                            │                      │          │
                  route_after_grading              │          │
                ┌───────────┴────────────┐         │          │
                ▼                        ▼         │          │
        score ≥ threshold         all chunks       │          │
        for ≥1 chunk              irrelevant       │          │
                │                        │         │          │
                │                        ▼         │          │
                │              ┌──────────────────┐│          │
                │              │   web_fallback   ││          │
                │              │ (DuckDuckGoSearch)│         │
                │              └──────────┬───────┘│          │
                │                         │        │          │
                └─────────────────────────┴────────┴──────────┘
                                          │
                                          ▼
                                  back to generate_response
```

### Node Reference

| Node | Type | Purpose |
|---|---|---|
| `generate_response` | LLM | Grok-4 fast-reasoning via xAI endpoint, bound to `tools_list`. Decides whether to call a tool or answer directly. |
| `tools` | `ToolNode` | Executes any tool the LLM calls (`calculator`, `get_stock_price`, `rag_tool`, `purchase_stock`). |
| `check_each_file` | Function | Pulls chunks from the most recent `rag_tool` ToolMessage and grades each one (1-10) using `gpt-4o-mini` with structured output (`ChunkScore`). |
| `web_fallback` | Function | Runs a DuckDuckGo search and injects the result as a `SystemMessage` for the LLM to use on the next pass. |

### Edges

- `START → generate_response`
- `generate_response → tools | END` via `tools_condition` (LangGraph prebuilt — routes on whether the LLM emitted tool calls)
- `tools → check_each_file` if the last tool was `rag_tool`, else `tools → generate_response` (`route_after_tools`)
- `check_each_file → generate_response` if at least `MIN_RELEVANT_CHUNKS` (1) scored ≥ `RELEVANCE_THRESHOLD` (6.0); else `→ web_fallback` (`route_after_grading`)
- `web_fallback → generate_response`

### State Schema

```python
class chatstate(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
```

### Corrective-RAG Grader

`grade_chunk` (in [main.py](main.py)) returns a `ChunkScore`:
```python
class ChunkScore(TypedDict):
    chunk_index: int
    score: float
    original_chunk: str
    reason: str
```
The aggregated list is stuffed into a `SystemMessage` (`chunk_scores=[...]`) so the routing function can parse and decide. This is also why `app.py` filters streamed tokens to only those originating from `langgraph_node == 'generate_response'` — otherwise grader JSON would leak into the UI.

## Tools

| Tool | Description |
|---|---|
| `calculator` | Add/sub/mul/div on two numbers. |
| `get_stock_price` | Tavily-backed stock lookup (recent days, includes answer). |
| `rag_tool` | FAISS similarity search; returns content + chunk artifact for grading. |
| `purchase_stock` | Human-in-the-loop interrupt for confirmation before "buying". |
| `decision_maker` | Demo decision tool (not registered in `tools_list`). |

## Memory

`SqliteSaver` persists checkpoints in `chatbot.db` keyed by `thread_id` (default `chat_thread_1`). Conversations survive restarts.

## Notes

- Document upload in `app.py` saves the file under `documents/` and calls `add_to_vectorstore`, which appends to the existing FAISS index or creates one if missing.
- `RELEVANCE_THRESHOLD` and `MIN_RELEVANT_CHUNKS` in [main.py](main.py) tune how aggressive the web fallback is.
- The main-LLM uses xAI Grok via the OpenAI-compatible endpoint; the grader/evaluator uses OpenAI `gpt-4o-mini`.
