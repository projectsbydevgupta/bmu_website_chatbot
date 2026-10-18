# 🎓 BMU Website Chatbot (CampusGPT)
### Enterprise Multi-Agent RAG with Microservices, JEV Guardrails & Benchmark Observatory

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq Cloud](https://img.shields.io/badge/Groq-LPU%20Inference-F55036.svg)](https://groq.com/)
[![Ollama Local](https://img.shields.io/badge/Ollama-Local%20Llama%203.2-000000.svg)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Cloud%20Vector%20Store-FF4F00.svg)](https://www.trychroma.com/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Tracing-4053D6.svg)](https://opentelemetry.io/)
[![Presentation](https://img.shields.io/badge/Presentation-Executive%20Slide%20Deck%20(PDF)-E25555.svg?logo=adobeacrobatreader&logoColor=white)](docs/BMU_Website_Chatbot_Presentation.pdf)

An institutional AI Chatbot tailored specifically for **BML Munjal University (BMU)**, indexing 1,200+ university pages. Engineered with a decoupled **microservices architecture**, an autonomous 3-agent **LangGraph** state machine, a two-stage **JEV Guardrails** security engine, dual-engine LLM runtime (**Groq Cloud LPU** vs. **Local Ollama**), and an interactive **100-Question Benchmark Observatory Dashboard** designed with **Google Stitch MCP**.

> 🌐 **Live Demo:** [https://bmu-website-chatbot.onrender.com/dashboard/](https://bmu-website-chatbot.onrender.com/dashboard/)  
> 📊 **Evaluation Observatory:** [https://bmu-website-chatbot.onrender.com/dashboard/evaluation.html](https://bmu-website-chatbot.onrender.com/dashboard/evaluation.html)  
> 📑 **Executive Presentation:** [BMU Website Chatbot Slide Deck (PDF)](docs/BMU_Website_Chatbot_Presentation.pdf)  
> 🧠 **Comprehensive System Architecture & Pure Logic:** For low-level mathematical derivations, full state graph logic, telemetry spans, and runbooks, see **[Brain.md](Brain.md)**.

---

## ⚡ End-to-End System Pipeline

```text
                       ┌────────────────────────────────────────────────────────┐
                       │                     User Question                      │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                                                   ▼
                       ┌────────────────────────────────────────────────────────┐
                       │            1. JEV Input Security Guardrails            │
                       │    (Jailbreak, Prompt Injection, Domain Scope Gate)    │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                                                   ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 2. LangGraph 3-Agent State Machine Workflow                                   │
│                                                                                                               │
│   ┌───────────────────────┐      ┌──────────────────────────────┐      ┌──────────────────────────────────┐   │
│   │   Agent 1: Query      │ ───► │  Hybrid Retrieval Tool       │ ───► │  Agent 2: Context Auditor        │   │
│   │   Specialist          │      │  ChromaDB + ONNX MiniLM      │      │  (Autonomous Self-Correction)    │   │
│   └───────────────────────┘      └──────────────────────────────┘      └─────────────────┬────────────────┘   │
│                                                                                          │                    │
│                                                                                          ▼                    │
│   ┌───────────────────────────────────────────────────────────┐        ┌─────────────────┴────────────────┐   │
│   │   Verified Answer with Page-Level References              │ ◄───── │  Agent 3: Response Synthesizer   │   │
│   │   (Rendered with Rich Typography & Provenance Badges)     │        │  (Groq LPU Cloud / Local Ollama) │   │
│   └───────────────────────────────────────────────────────────┘        └──────────────────────────────────┘   │
└──────────────────────────────────────────────┬────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                       ┌────────────────────────────────────────────────────────┐
                       │         3. OpenTelemetry & Arize Phoenix Tracing       │
                       │    (Span latency, token usage, Benchmark Observatory)  │
                       └────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Architectural Highlights

- **3-Agent LangGraph State Machine:**
  - **Agent 1 (Query Specialist):** Intent classification and entity keyword expansion.
  - **Deterministic Hybrid Retrieval:** Re-ranks candidates ($0.68 \times \text{Semantic} + 0.32 \times \text{BM25 Keyword}$) across 1,200+ BMU webpages with a 3.0s live-scrape safeguard.
  - **Agent 2 (Context Auditor):** Evaluates factual evidence sufficiency with autonomous self-correction loops (inspects fallback sources if primary context lacks specific details like Chancellor designations).
  - **Agent 3 (Response Synthesizer):** Synthesizes grounded, citation-backed answers strictly from vetted context.
- **Dual-Stage JEV Security Guardrails:** Pre-screens against prompt injections, DAN jailbreaks, toxic inputs, and out-of-scope queries; post-validates grounding before returning answers.
- **Zero-PyTorch & Under 512MB RAM Footprint:** Replaced PyTorch with an **ONNX Runtime MiniLM** pipeline, reducing RAM usage by 70% (~150MB active RAM) to ensure stable deployment on free-tier cloud instances (Render).
- **Dual LLM Runtime (Cloud & Local):** Instant client-side switching between **Groq Cloud LPU** (`gpt-oss-20b`) and **Local Ollama** (`llama-3.2`), complete with visual source provenance badges.
- **Mobile-First Responsive Web Design:** Native app-like `100dvh` viewport, pinned bottom input bar with iPhone safe-area insets, horizontal swipeable suggestion pills, and safe markdown rendering.
- **100-Question Evaluation Observatory:** Interactive dashboard evaluating 20 academic categories across 5 RAGAS mathematical dimensions (Relevance, Faithfulness, Completeness, Conciseness, Correctness).

### 📊 Benchmark Performance Summary

| Metric | Measured Value | Evaluation & System Context |
| :--- | :--- | :--- |
| **Top-1 Retrieval Accuracy** | **81.0%** | Evaluated on 100-question institutional BMU test suite |
| **ROC-AUC Score** | **0.87** | Semantic vs. keyword separation quality |
| **Cloud Inference Latency** | **~1.8s** | Groq LPU (`gpt-oss-20b`) end-to-end multi-agent execution |
| **Active Memory Footprint** | **~150MB** | Zero-PyTorch ONNX Runtime (70% memory reduction, <512MB cap) |
| **Guardrail Evaluation Speed** | **< 3ms** | Deterministic pre-screen injection & safety gate |

---

## 🛠️ Microservices Architecture

| Microservice | Port | Primary File | Core Responsibility |
| :--- | :--- | :--- | :--- |
| **Backend Gateway** | `8000` | `semantic.py` | FastAPI gateway, JEV input guardrails, LangGraph orchestration, static UI server |
| **Retrieval Engine** | `8001` | `retrieval_service.py` | Isolated ONNX MiniLM embeddings, ChromaDB Cloud search, hybrid 68/32 re-ranking |
| **Ingestion Engine** | `8002` | `ingestion_service.py` | Bounded BFS web crawler, Trafilatura scraper, SHA-256 deduplication, Voyage AI vectors |
| **Observability** | `6006` | `phoenix_service.py` | Arize Phoenix OpenTelemetry tracing, span trees, token & latency telemetry |
| **Frontend UI** | `/dashboard/` | `frontend/` | Responsive Chat Assistant & Obsidian Telemetry Observatory |

---

## 🔌 API Endpoint Reference

### `POST /ask`
Send a query to the multi-agent RAG pipeline:
```bash
curl -X POST "https://bmu-website-chatbot.onrender.com/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the BTech fees?",
    "llm_provider": "groq",
    "enable_guardrails": true
  }'
```

**Response Payload:**
```json
{
  "query": "What are the BTech fees?",
  "answer": "The tuition component for BTech (CSE) is ₹1,640,000 (Domestic) / $28,400 (Foreign/OCI)...",
  "top_results": [{"url": "https://www.bmu.edu.in/fee-structure/", "score": 0.651}],
  "guardrail": {"status": "passed", "action": "allow"},
  "agent_trace": [{"agent": "Agent 1 (QuerySpecialist)", "latency_ms": 320.5}],
  "llm_provider": "groq",
  "guardrails_enabled": true
}
```

---

## 📦 Quickstart & Local Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/projectsbydevgupta/bmu_website_chatbot.git
cd bmu_website_chatbot
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=default_tenant
CHROMA_DATABASE=default_database
LOCAL_LLM_URL=http://localhost:11434
```

### 3. Run All Services
- **Windows (One-Click Launcher):**
  ```powershell
  .\run.bat
  ```
- **Linux / Cloud Deployments (Render):**
  ```bash
  bash start.sh
  ```
Access the application at `http://127.0.0.1:8000/dashboard/`.

---

## 📂 Repository File Structure

```text
├── semantic.py           # FastAPI Gateway & Orchestrator (:8000)
├── retrieval_service.py  # ONNX Hybrid Retrieval Engine (:8001)
├── ingestion_service.py  # BFS Web Scraper & Voyage AI Ingestion (:8002)
├── phoenix_service.py    # Arize Phoenix OpenTelemetry Server (:6006)
├── bmu_rag_graph.py      # LangGraph 3-Agent State Machine
├── jev_guardrails.py     # Dual-Tier Security & Grounding Guardrails
├── run.bat               # Windows Multi-Process One-Click Launcher
├── start.sh              # Cloud Container Entrypoint (Render)
├── frontend/             # Responsive Chat UI & Stitch Observatory
└── Brain.md              # Master Pure Logic & Architectural Runbook
```

---

## 📚 Complete Technical Documentation

For complete mathematical formulations, system topology diagrams, test suites, and internal logic:
👉 **Read the comprehensive reference manual: [Brain.md](Brain.md)**

