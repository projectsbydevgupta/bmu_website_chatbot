# 🎓 BMU Website Chatbot: Enterprise Multi-Agent RAG System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)](https://bmu-website-chatbot.onrender.com)
[![Observatory](https://img.shields.io/badge/Benchmark%20Observatory-Live-6366F1.svg?style=for-the-badge&logo=google-cloud&logoColor=white)](https://bmu-website-chatbot.onrender.com/dashboard/evaluation.html)
[![Arize Phoenix](https://img.shields.io/badge/Arize%20Phoenix-Tracing%20Cloud-FF007A.svg?style=for-the-badge&logo=opentelemetry&logoColor=white)](https://app.phoenix.arize.com/s/bmu-chatbot/projects)
[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-Passing-10B981.svg?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/projectsbydevgupta/bmu_website_chatbot/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00.svg?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)

An enterprise-grade, institutional AI Chatbot tailored specifically for **BML Munjal University (BMU)**. Powered by an autonomous 3-agent **LangGraph** state machine, a decoupled **4-microservices architecture**, a two-stage **JEV Guardrails** security and hallucination mitigation engine, a **1,024-dimensional Voyage AI vector knowledge base** on ChromaDB Cloud with real-time scraping fallback, dual LLM runtime (**Groq Cloud LPU** vs. **Local Ollama**), and end-to-end **Arize Phoenix** OpenTelemetry distributed tracing.

---

### 🌐 Live Production Links

| Service | Live URL | Description |
| :--- | :--- | :--- |
| 🚀 **Web Chat Assistant** | [bmu-website-chatbot.onrender.com](https://bmu-website-chatbot.onrender.com) | Production conversational AI interface with dual-engine toggle & guardrails |
| 📊 **Benchmark Observatory** | [bmu-website-chatbot.onrender.com/dashboard/evaluation.html](https://bmu-website-chatbot.onrender.com/dashboard/evaluation.html) | 100-Query RAGAS telemetry dashboard with radar plots & side-by-side inspection |
| 🔭 **Arize Phoenix Tracing** | [app.phoenix.arize.com/s/bmu-chatbot/projects](https://app.phoenix.arize.com/s/bmu-chatbot/projects) | Cloud OpenTelemetry tracing: spans, latencies, tokens, and prompt inspector |
| 🩺 **System Health Probe** | [bmu-website-chatbot.onrender.com/health](https://bmu-website-chatbot.onrender.com/health) | Real-time JSON health status across all microservices and vector indices |
| 📁 **GitHub Repository** | [github.com/projectsbydevgupta/bmu_website_chatbot](https://github.com/projectsbydevgupta/bmu_website_chatbot) | Official source repository with automated GitHub Actions CI/CD pipeline |

---

## 🏛️ System Architecture

```text
                                 [ User / Student Browser ]
                                              │
                      HTTP POST /ask { query, llm_provider, guardrails_enabled }
                                              │
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  PORT 8000: BACKEND ORCHESTRATOR (semantic.py)                                         │
│                                                                                        │
│  1. JEV INPUT GUARDRAILS (jev_guardrails.py)                                           │
│     ├── Prompt Injection / Jailbreak Filter (DAN, override patterns)                   │
│     ├── Toxicity & Abuse Detection (Civility enforcement)                              │
│     └── Academic Scope Classifier (Filters non-BMU queries; bypassable via UI toggle)  │
│                                                                                        │
│  2. LANGGRAPH 3-AGENT STATE MACHINE (bmu_rag_graph.py)                                 │
│     ├── Agent 1: Query Specialist ── Intent classification & academic acronym expansion│
│     │            │                                                                     │
│     │            ▼                                                                     │
│     │     Hybrid Retrieval Tool ── Vector Search (Port 8001) + 3.0s Live Scrape Fallback│
│     │            │                                                                     │
│     │            ▼                                                                     │
│     ├── Agent 2: Context Auditor ── Sufficiency verification & fallback candidate loop │
│     │            │                                                                     │
│     │            ▼                                                                     │
│     └── Agent 3: Response Synthesizer ── Fact-grounded markdown & verified citations   │
│                                                                                        │
│  3. JEV OUTPUT GUARDRAILS                                                              │
│     └── Entity containment check against retrieved context + disclaimer injection      │
└──────────────────────────┬───────────────────────────────┬─────────────────────────────┘
                           │                               │
                           ▼                               ▼
┌──────────────────────────────────────┐       ┌────────────────────────────────────────┐
│ PORT 8001: RETRIEVAL MICROSERVICE    │       │ PORT 6006: ARIZE PHOENIX TRACING       │
│ (retrieval_service.py)               │       │ (phoenix_service.py + phoenix_tracer)  │
│ ├── Voyage AI 1024-d / MiniLM 384-d  │       │ ├── OpenTelemetry Distributed Spans    │
│ ├── ChromaDB Cloud Vector Store      │       │ ├── Multi-Agent DAG Visualizer         │
│ └── Hybrid Score: 0.68*Cos + 0.32*KW │       │ └── Latency, Token & Cost Breakdown   │
└──────────────────────────────────────┘       └────────────────────────────────────────┘
                           ▲
                           │ Background Web Crawling & Incremental Voyage Vectorization
┌──────────────────────────┴───────────┐
│ PORT 8002: INGESTION MICROSERVICE    │
│ (ingestion_service.py)               │
│ ├── Trafilatura BMU Web Scraper      │
│ ├── Extractive Summarizer (BM25+TF)  │
│ └── SHA-256 Deduplicated Cloud Index │
└──────────────────────────────────────┘
```

---

## ⚡ Core Engineering Pillars

### 1. Decoupled 4-Microservices Architecture
To prevent heavy ML dependencies (PyTorch, ONNX, ChromaDB) from degrading HTTP throughput, the system isolates concerns into four distinct services:

| Port | Service | Entry File | Core Responsibility |
| :---: | :--- | :--- | :--- |
| **8000** | **Backend Orchestrator** | `semantic.py` | FastAPI gateway, session state, JEV guardrails, LangGraph runner, UI serving |
| **8001** | **Retrieval Microservice** | `retrieval_service.py` | Isolated vector similarity search, hybrid cosine + keyword scoring |
| **8002** | **Ingestion Microservice** | `ingestion_service.py` | Trafilatura web crawling, BM25+TF-IDF extractive summarization, Voyage indexing |
| **6006** | **Arize Phoenix Server** | `phoenix_service.py` | OpenTelemetry distributed tracing server and real-time observability UI |

### 2. Autonomous 3-Agent LangGraph State Machine
Traditional single-prompt RAG suffers from vocabulary mismatches and context hallucination. Our stateful LangGraph engine partitions cognitive load across three specialized nodes:
* **Agent 1 (Query Specialist):** Classifies student intent (`admissions`, `fees`, `placements`, `faculty`, `hostel`) and expands conversational queries into comprehensive academic search terms (e.g., *"SOM"* $\rightarrow$ *"School of Management"*).
* **Deterministic Hybrid Retrieval:** Queries ChromaDB Cloud with a 3.0s timeout safeguard on live URLs, preparing 3 distinct candidates (Top-Ranked Document, Secondary Document, and Synthesized Multi-Document Cluster).
* **Agent 2 (Context Auditor):** Evaluates candidate context for factual sufficiency (`SUFFICIENT`, `TRY_FALLBACK`, or `NOT_FOUND`). If incomplete, it loops to alternative candidates before synthesis.
* **Agent 3 (Response Synthesizer):** Generates structured Markdown answers strictly grounded in the audited context. For broad queries (e.g. *"Faculty details"*, *"Courses offered"*), it executes institutional multi-school synthesis, preventing single-department bias.

### 3. Two-Stage JEV Guardrails (Judge / Evaluate / Validate)
* **Input Defense:** Pre-inference regex and semantic classifiers detect adversarial prompt injections (DAN, jailbreaks, system leaks), toxicity/harassment, and out-of-scope non-BMU queries.
* **Output Defense:** Post-inference containment scoring measures entity overlap between the response and source context, flagging ungrounded claims.
* **Bypass Mode:** Users can toggle `🛡️ Guardrails: OFF` from the UI header to query the underlying model directly without domain constraints.

### 4. Vector Database & Dual Retrieval Strategy
ChromaDB Cloud (`api.trychroma.com`) hosts two isolated collections to avoid dimension collisions:
* **`bmu_voyage_knowledge` (Primary Production):** 1,024-dimensional embeddings via Voyage AI (`voyage-4-large`). Supports incremental indexing with SHA-256 chunk deduplication.
* **`bmu_summaries` (Lightweight Fallback):** 384-dimensional ONNX embeddings (`all-MiniLM-L6-v2`) providing zero-cloud-cost fallback search.
* **Hybrid Ranking Formula:**
  $$\text{Score} = (0.68 \times \text{Cosine Similarity}) + (0.32 \times \text{Keyword Overlap})$$

### 5. Dual LLM Runtime
* **☁️ Cloud (Groq LPU):** Powered by `openai/gpt-oss-120b` (Synthesis) and `gpt-oss-20b` (Specialist) delivering sub-second median latency (~850ms–2,100ms).
* **💻 Local (Ollama):** Powered by `llama3.2:3b` for 100% offline, privacy-first, zero-cost inference.
* Toggled dynamically via the header switch without restarting backend sessions.

### 6. End-to-End Observability (Arize Phoenix)
Every transaction emits an OpenTelemetry span hierarchy:
* Trace Root (`/ask`) captures overall latency, provider, and guardrail flags.
* Child spans record prompt tokens, completion tokens, latency waterfalls, and internal agent decisions.
* Viewable locally at `http://localhost:6006` or on [Arize Phoenix Cloud](https://app.phoenix.arize.com/s/bmu-chatbot/projects).

---

## 📊 100-Question Benchmark & RAGAS Observatory

The project includes an automated evaluation harness ([`benchmark_100_questions.py`](benchmark_100_questions.py)) spanning 20 university categories (Admissions, Fees, Placements, Hostels, Adversarial, Out-of-Scope) scored across five standard RAGAS dimensions:
1. **Relevance:** Query-to-answer semantic and keyword alignment ($0.0 - 5.0$).
2. **Faithfulness:** Response containment against retrieved source text.
3. **Completeness:** Key factual entity extraction against gold-standard ground truth.
4. **Conciseness:** Information density without redundant boilerplate.
5. **Correctness:** Semantic cosine similarity against verified academic answers.

Interactive results are rendered live in the **Benchmark Observatory** ([evaluation.html](https://bmu-website-chatbot.onrender.com/dashboard/evaluation.html)), featuring radar plots, latency distributions, confusion matrices, and side-by-side Groq vs. Ollama comparisons.

---

## 🛠️ Installation & Local Setup

### Prerequisites
* **Python 3.10+** (Tested on Python 3.11 & 3.14)
* **Git**
* *(Optional)* **Ollama** with `llama3.2:3b` for local offline inference:
  ```bash
  ollama pull llama3.2:3b
  ```

### Step 1: Clone Repository
```bash
git clone https://github.com/projectsbydevgupta/bmu_website_chatbot.git
cd bmu_website_chatbot
```

### Step 2: Environment Configuration
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Key configuration values:
```env
# Cloud LLM (Groq)
GROQ_API_KEY=gsk_your_groq_api_key

# Vector Database (ChromaDB Cloud)
CHROMA_CLOUD_API_KEY=your_chroma_api_key
CHROMA_CLOUD_TENANT=default_tenant
CHROMA_CLOUD_DATABASE=BMU_WEBSITE_CHATBOT

# Primary Embeddings (Voyage AI)
VOYAGE_API_KEY=pa-your_voyage_key
RETRIEVAL_ENGINE=voyage

# Observability (Arize Phoenix Cloud)
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com
PHOENIX_API_KEY=your_phoenix_api_key
```

### Step 3: Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 4: Run Microservices
**Option A: One-Click Launcher (Windows)**
```cmd
run.bat
```
*(Automatically launches all 4 microservices in separate background processes and opens the browser).*

**Option B: Manual Startup**
```bash
# Terminal 1: Arize Phoenix Tracing
python phoenix_service.py

# Terminal 2: Retrieval Microservice
python retrieval_service.py

# Terminal 3: Ingestion Microservice
python ingestion_service.py

# Terminal 4: Backend Orchestrator
python semantic.py
```
Open **`http://localhost:8000/dashboard/`** in your browser.

---

## 🧪 CI/CD Pipeline & Quality Gates

The project enforces strict automated verification before every commit via [`test_ci_pipeline.py`](test_ci_pipeline.py) and GitHub Actions (`.github/workflows/ci_pipeline.yml`):
* **Suite 1:** Environment & Configuration Integrity
* **Suite 2:** Retrieval Microservice Vector Search (Voyage 1024-d & MiniLM 384-d)
* **Suite 3:** Ingestion Microservice Health & Crawl API Contracts
* **Suite 4:** JEV Input Guardrails (DAN Jailbreaks, Toxicity, Scope Filtering)
* **Suite 5:** LangGraph Multi-Agent RAG Graph Execution & Context Auditing
* **Suite 6:** Dual LLM Provider Routing (Groq Cloud & Local Ollama)
* **Suite 7:** JEV Output Guardrail Grounding & Safety Verification

Run test suite locally:
```bash
python test_ci_pipeline.py
```

---

## 📡 API Reference

| Endpoint | Method | Service | Description |
| :--- | :---: | :---: | :--- |
| `/ask` | `POST` | Orchestrator (8000) | Primary student chat endpoint with guardrails & multi-agent execution |
| `/health` | `GET` | Orchestrator (8000) | Live health probe reporting status of downstream microservices |
| `/agents/status` | `GET` | Orchestrator (8000) | Real-time state of Agent 1, Agent 2, Agent 3, and JEV Guardrails |
| `/evaluation/summary`| `GET` | Orchestrator (8000) | Aggregated KPIs and RAGAS metric averages |
| `/evaluation/results`| `GET` | Orchestrator (8000) | Granular 100-question benchmark records |
| `/retrieve` | `POST` | Retrieval (8001) | Hybrid cosine + keyword vector search over ChromaDB Cloud |
| `/retrieve/compare` | `POST` | Retrieval (8001) | Side-by-side comparison of 1024-dim Voyage vs. 384-dim MiniLM |
| `/ingest/crawl` | `POST` | Ingestion (8002) | Crawls BMU URLs, extracts text, and vectorizes into ChromaDB |

---

## 📄 License & Institutional Attribution

Engineered for **BML Munjal University (BMU)**.  
Developed by **[Dev Gupta](https://github.com/projectsbydevgupta)**.  
Open-source under the [MIT License](LICENSE).
