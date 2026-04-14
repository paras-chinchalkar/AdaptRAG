# 🚀 Amazing Adaptive RAG

An enterprise-grade, intelligent Retrieval-Augmented Generation (RAG) system powered by **LangGraph**, **Groq (Llama-3.1-8b)**, and **Streamlit**. 

Unlike standard linear RAG pipelines, this system dynamically routes between vector database lookups and real-time web searches, self-corrects hallucinations, and intelligently rewrites complex queries to ensure the highest quality answers.

---

## ✨ Advanced Features Implementation

This project implements state-of-the-art Adaptive RAG features natively via LangGraph:

### 1. 🧭 Intelligent Routing
The router LLM analyzes user queries and decides whether to route them to a **Vector Store** (for domain-specific knowledge like prompt engineering or agent setups) or directly to **Web Search** (for general or real-time queries).

### 2. 🔍 Ensemble Hybrid Retrieval
Instead of relying purely on FAISS (semantic) similarity, this project combines it with **BM25 Keyword Retrieval**. This hybrid approach (weighted 60% Semantic, 40% Keyword) guarantees that both overall meaning and exact terminologies are caught.

### 3. 🛡️ Hallucination & Relevance Grading
Every piece of retrieved context and generated text is graded:
* **Document Grader:** Strips out irrelevant context chunks. If all retrieved docs are bad, it seamlessly triggers a web search.
* **Hallucination Grader:** Checks if the generated answer is strictly grounded in the provided facts.
* **Answer Grader:** Checks if the grounded answer directly resolves the original user query.

### 4. 🔄 Active Query Rewriting
If initial retrievals or web searches fail to yield useful information, the `Query Rewriter` node activates. It reasons about the underlying semantic intent of the initial question and reformulates it into an optimized search query before falling back to search.

### 5. 🛑 Robust Self-Correction (Max Retries)
The `GraphState` preserves a strict mathematical `retries` int. If the LLM enters a hallucination loop or fails to answer the question, the retry mechanism catches it and safely terminates the graph after 3 iterations—shielding your API rate limits from infinite cycles.

### 6. ⚡ Blazing Fast Cold Starts
Document loaders (`WebBaseLoader`) and HF Embeddings (`all-mpnet-base-v2`) are aggressively cached (`docs_cache.pkl` and `./faiss_cache`). Startup latency has been reduced from minutes to sub-second parsing.

---

## 🛠️ Architecture Workflow

The underlying generic workflow state chart implemented in `src/rag/graph.py` is dynamically managed via `StateGraph`:

1.  **Entry Strategy:** `route_question` \-\> determines if `websearch` or `retrieve`.
2.  **Retrieve:** Hybrid BM25/FAISS \-\> `grade_documents`.
3.  **Grade:** Decides if context is useful. If NOT, routes to `rewrite` query node.
4.  **Rewrite:** Rewrites query \-\> `websearch`.
5.  **Generate:** Synthesizes answer.
6.  **Verify & Loop:** `grade_generation` verifies factual grounding. It loops back to `rewrite` / `generate` or cleanly breaks out to `END` on hitting `max_retries`.

---

## 💻 Tech Stack

*   **Workflow Engine:** LangGraph
*   **LLM Provider:** Groq (`llama-3.1-8b-instant`) strictly enforced in `json_mode` to prevent Pydantic tool parse errors.
*   **Vector Database:** FAISS
*   **Web Search:** DuckDuckGo Search
*   **Embeddings:** HuggingFace (`sentence-transformers`)
*   **Keyword Retrieval:** `rank_bm25`
*   **Frontend UI:** Streamlit

---

## 📦 Setup & Quickstart

1. **Clone & Install Dependencies**
```bash
pip install -r requirements.txt
```
*(Make sure to also install `rank_bm25` if not in requirements!)*

2. **Set Environment Variables**
Configure your `.env` file at the root of the project with your API keys:
```env
GROQ_API_KEY="gsk_your_groq_api_key_here"
```

3. **Run the Interface**
Execute the Streamlit application to chat with the fully graphical RAG agent:
```bash
streamlit run src/ui/app.py
```

4. **Headless Execution (Testing)**
Run the native python graph workflow in terminal:
```bash
python test_graph.py
```
