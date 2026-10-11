# The Brain: Comprehensive Low-Level Architecture & Pure Logic Reference
## BMU Website Chatbot (Multi-Agent RAG, JEV Guardrails, Microservices & Phoenix Observability)

---

## 1. System Topology & Architectural Blueprint

The BMU Website Chatbot is a distributed, multi-service, multi-agent AI system designed to answer academic, admissions, fee, and administrative queries regarding BML Munjal University (BMU) with mathematical grounding and real-time observability.

### 1.1 Process & Network Port Map

| Component | Port | Technology | Primary File | Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **Frontend UI** | `3000` | Vanilla HTML5 / CSS3 / ES6 | `frontend/index.html`, `frontend/app.js` | User chat interface, agent latency visualizer, guardrail badge renderer |
| **Backend Orchestrator** | `8000` | FastAPI, Uvicorn, LangGraph | `semantic.py` | Receives client queries, executes JEV guardrails, runs LangGraph RAG graph |
| **Retrieval Microservice** | `8001` | FastAPI, SentenceTransformers, ChromaDB | `retrieval_service.py` | Isolated embedding inference, ChromaDB Cloud search, hybrid 68/32 re-ranking |
| **Ingestion Microservice** | `8002` | FastAPI, Trafilatura, Voyage AI | `ingestion_service.py`, `voyage_client.py` | Bounded web crawler, semantic chunker, SHA-256 deduplication, Voyage AI vector indexer |
| **Phoenix Tracing Server** | `6006` | Arize Phoenix, OpenTelemetry, Uvicorn | `phoenix_service.py`, `phoenix_tracer.py` | Collects OpenInference traces, displays flame graphs, latency and token metrics |

### 1.2 Inter-Service Data Flow Diagram

```
[ User Browser @ :3000 ]
           │
           │ HTTP POST /ask { "query": "..." }
           ▼
[ Backend Orchestrator @ :8000 ]
  ├── 1. JEV Input Guardrail (<3ms deterministic pre-screen)
  │      ├── Regex Injection & Toxicity Patterns
  │      └── Academic Scope Classifier (bmu_academic vs out_of_scope)
  │
  ├── 2. Phoenix Tracer OpenInference Spans (`bmu_rag_pipeline` CHAIN)
  │      ▼
  │   [ LangGraph StateGraph Workflow ]
  │      ├── Node 1: Agent 1 (QuerySpecialist) ───► [ Groq API Key 1: gpt-oss-20b ]
  │      │           (Entity expansion & keyword extraction)
  │      │
  │      ├── Node 2: Retrieval Tool (Deterministic)
  │      │           │
  │      │           ├── HTTP POST :8001/retrieve { "query": opt_q, "top_k": 8 }
  │      │           │    ▼
  │      │           │   [ Retrieval Microservice @ :8001 ]
  │      │           │    ├── Local all-MiniLM-L6-v2 (384-dim)
  │      │           │    ├── ChromaDB Cloud (bmu_summaries)
  │      │           │    └── Hybrid Rank = 0.68*Semantic + 0.32*Keyword
  │      │           │
  │      │           └── Multi-Candidate Extractor (Trafilatura Web Scraper)
  │      │                ├── Candidate 1: Live Scraped URL #1
  │      │                ├── Candidate 2: Live Scraped URL #2
  │      │                └── Candidate 3: Top-6 ChromaDB Summary Bundle
  │      │
  │      ├── Node 3: Agent 2 (ContextAuditor) ───► [ Groq API Key 2: gpt-oss-20b ]
  │      │           │   (Evaluates factual sufficiency of Candidate Context)
  │      │           ▼
  │      │      [ Conditional Edge: route_after_context_audit ]
  │      │           ├── SUFFICIENT ─────────────► Node 5: Agent 3 (ResponseSynthesizer)
  │      │           ├── TRY_FALLBACK ───────────► Node 4: advance_candidate (Self-Correction Loop)
  │      │           │                             (Increments pointer, loops back to Agent 2)
  │      │           └── NOT_FOUND ──────────────► Refusal Handler (Authoritative BMU Helpdesk)
  │      │
  │      └── Node 5: Agent 3 (ResponseSynthesizer) ──► [ Groq API Key 3: gpt-oss-120b ]
  │                  (Strictly context-grounded synthesis)
  │
  ├── 3. JEV Output Guardrail (<1ms deterministic post-screen)
  │      ├── Hallucination / Context Grounding check (overlap ratio * 1.2)
  │      └── Safety violation re-scan
  │
  └── 4. OpenTelemetry Protocol (OTLP/HTTP) ──► [ Arize Phoenix @ :6006/v1/traces ]
```

---

## 2. LangGraph Multi-Agent RAG Orchestration (`bmu_rag_graph.py`)

The multi-agent workflow is modeled as a state machine via LangGraph's `StateGraph`.

### 2.1 Complete State Schema Definitions

```python
class CandidateContext(TypedDict):
    source: str      # Exact URL or "Chroma Document #1" or "BMU Curated Knowledge Base"
    url: str         # Source URL
    content: str     # Raw scraped text (<3500 chars) or bundled summaries
    score: float     # Hybrid retrieval score (0.0 to 1.0)
    type: str        # "scraped_page" | "stored_summary"

class AgentTraceItem(TypedDict):
    agent: str              # Identifier (e.g. "Agent 1 (QuerySpecialist)")
    latency_ms: float       # Execution duration in milliseconds
    details: Dict[str, Any] # Step-specific metadata (models, keywords, decisions, reasons)

class BMURAGState(TypedDict):
    raw_query: str          # Original question submitted by user
    
    # Agent 1 outputs
    optimized_query: str    # Semantically enriched query for vector retrieval
    keywords: List[str]     # 3-5 extracted core keywords
    intent_category: str    # "admissions" | "courses" | "fees" | "hostels" | "placements" | "faculty" | "campus" | "general"
    
    # Retrieval Tool outputs
    candidates: List[CandidateContext] # Pre-fetched extraction candidates (up to 3)
    current_candidate_idx: int         # Pointer tracking which candidate Agent 2 is inspecting
    
    # Agent 2 outputs
    audit_decision: Literal["SUFFICIENT", "TRY_FALLBACK", "NOT_FOUND"]
    audit_reason: str       # Justification sentence from auditor
    vetted_context: str     # Approved text payload passed to Agent 3
    
    # Agent 3 outputs
    final_answer: str       # Natural language answer generated for student
    top_results: List[Dict[str, Any]] # Top URLs with similarity scores
    
    # Observability
    agent_trace: List[AgentTraceItem] # Chronological trace sequence for UI & Phoenix
    error: Optional[str]
```

---

### 2.2 Node 1: Agent 1 - Query Specialist (`query_specialist_node`)

#### Pure Logic & Responsibility
Translates ambiguous, acronym-heavy student queries into dense, entity-rich queries suitable for vector retrieval, and identifies intent categories.

- **Groq Client & Configuration:**
  - Env Var: `GROQ_API_KEY_QUERY_AGENT` (fallback: `GROQ_API_KEY`)
  - Model: `openai/gpt-oss-20b`
  - Temperature: `0.1` (low variance, near-deterministic)
  - Max Tokens: `600`
- **System Prompt:**
  `"You are a specialized academic query optimizer for BML Munjal University. Output strictly valid JSON."`
- **Transformation Prompt Logic:**
  1. Enforces acronym expansion: `B.Tech` -> `B.Tech Computer Science and Engineering CSE`, `BMU` -> `BMU BML Munjal University`, `fees` -> `fees tuition fee structure`.
  2. Strips conversational conversational fluff (`"Can you tell me..."`, `"I was wondering..."`).
  3. Formulates a JSON schema response:
     ```json
     {
       "optimized_query": "...",
       "keywords": ["...", "..."],
       "intent": "admissions | courses | fees | hostels | placements | faculty | campus | general"
     }
     ```
- **Robust JSON Extraction (`_safe_json_parse`):**
  Uses regex `r"```(?:json)?\s*(\{.*?\})\s*```"` to unpack code fences, falling back to `r"(\{.*\})"` for unformatted JSON, and parsing via `json.loads()`.
- **Heuristic Deterministic Fallback (Fault Tolerance):**
  If Groq fails, returns empty, or errors out:
  1. Stopword filter: Strips common English stopwords (`what`, `are`, `the`, `and`, `for`, `with`, `from`, `at`, `is`, `in`, `of`, `to`, `how`, `can`, `does`, `which`).
  2. Extracts terms of length > 2 as keywords.
  3. Intent matching rules:
     - Contains `fee`, `fees`, `cost`, `scholarship` -> `"fees"`
     - Contains `eligibility`, `admission`, `criteria`, `apply` -> `"admissions"`
     - Contains `b.tech`, `cse`, `mba`, `course`, `program` -> `"courses"`
     - Otherwise -> `"general"`

---

### 2.3 Node 2: Multi-Candidate Retrieval & Extraction Tool (`retrieval_tool_node`)

#### Pure Logic & Responsibility
Retrieves candidate documents, performs hybrid ranking, and constructs candidate contexts.

1. **Primary Retrieval Path (HTTP Microservice):**
   - Sends HTTP POST to `http://127.0.0.1:8001/retrieve` with `{"query": optimized_query, "top_k": 8}` and a 10-second timeout.
   - Extracts ranked results: `[{"summary": ..., "url": ..., "score": ...}, ...]`.
2. **In-Process Resilient Fallback:**
   - If the microservice is down, unreachable, or times out, it switches `retrieval_mode` to `"in_process_fallback"`.
   - Lazily loads local `SentenceTransformer("all-MiniLM-L6-v2")` and `chromadb.CloudClient`.
   - Embeds query: `query_emb = embedder.encode([opt_query]).tolist()`.
   - Queries collection `bmu_summaries` for 8 nearest neighbors.
   - Calculates hybrid score for each document:
     $$\text{HybridScore} = 0.68 \times (1.0 - \text{distance}) + 0.32 \times \text{KeywordScore}$$
3. **Threshold Gate (`0.10`):**
   - If `top_score < 0.10`, the query is flagged as low similarity, skipping live web scraping to prevent wasted network I/O.
4. **Candidate Context Construction Pipeline:**
   - **Candidate 1 & 2 (Live Web Pages):**
     - Extracts the URL from the top 2 ranked documents.
     - Uses `trafilatura.fetch_url(url)` followed by `trafilatura.extract(downloaded)`.
     - Boilerplate removal: Trafilatura strips navigation bars, footers, headers, ads, and sidebars.
     - If extracted text length > 100 chars, truncates to 3,500 characters and labels as `"scraped_page"`.
     - If the fetch fails or the page returns a 404/403/captcha, it falls back to the stored ChromaDB summary (`"stored_summary"`).
   - **Candidate 3 (Synthesized Knowledge Bundle):**
     - Merges the summaries of the top 6 ChromaDB documents into a unified multi-document context bundle separated by `---` delimiters.
     - Serves as the ultimate fallback if specific scraped pages are missing details or broken.

---

### 2.4 Node 3: Agent 2 - Context Auditor (`context_auditor_node`)

#### Pure Logic & Responsibility
Validates factual sufficiency of the current candidate context against the user question before passing it to synthesis.

- **Groq Client & Configuration:**
  - Env Var: `GROQ_API_KEY_CONTEXT_AGENT` (fallback: `GROQ_API_KEY`)
  - Model: `openai/gpt-oss-20b`
  - Temperature: `0.1`
  - Max Tokens: `600`
- **System Prompt:**
  `"You are the BMU Context Relevance Auditor. Output strictly valid JSON."`
- **Decision Taxonomy:**
  1. `SUFFICIENT`: The context directly and factually answers the question.
  2. `TRY_FALLBACK`: The context lacks the answer, but further candidate sources remain uninspected.
  3. `NOT_FOUND`: The context proves the information does not exist or all sources are exhausted.
- **Deterministic Loop Termination:**
  - If `is_last_candidate` (i.e., `idx >= len(candidates) - 1`) and the verdict is `TRY_FALLBACK`, the node forces a final decision:
    - If keyword overlap >= 2 words and context length > 300 chars: upgrades to `SUFFICIENT` (`"Conclusive acceptance on final candidate bundle"`).
    - Otherwise: changes verdict to `NOT_FOUND` (`"Exhausted all candidate sources"`).

---

### 2.5 Conditional Routing & Autonomous Self-Correction Loop

#### `route_after_context_audit(state: BMURAGState) -> str`
```python
decision = state.get("audit_decision", "NOT_FOUND")
if decision == "SUFFICIENT":
    return "synthesize"          # Routes to response_synthesizer_node
elif decision == "TRY_FALLBACK":
    return "advance_candidate"    # Routes to advance_candidate_node (loop)
return "handle_not_found"         # Routes to handle_not_found_node
```

#### Self-Correction Loop Math
1. `advance_candidate_node`:
   $$\text{new\_idx} = \text{current\_candidate\_idx} + 1$$
2. Edge: `advance_candidate` points back to `context_auditor`.
3. The graph loops dynamically:
   - Candidate 1 (Scraped URL 1) -> `TRY_FALLBACK`
   - Candidate 2 (Scraped URL 2) -> `TRY_FALLBACK`
   - Candidate 3 (Chroma Summary Bundle) -> `SUFFICIENT` -> `response_synthesizer`
   - Or if all fail -> `NOT_FOUND` -> `handle_not_found`.

---

### 2.6 Node 4: Authoritative Not Found Handler (`handle_not_found_node`)

Executes when information is confirmed missing in BMU records:
- Returns an official referral rather than a generic model hallucination:
  `"I could not find sufficiently relevant information in official BML Munjal University records for this query. Please check the official BMU portal (https://www.bmu.edu.in) or contact the admissions office at info@bmu.edu.in / 1800-103-6888."`
- Sets `agent_trace` action to `"Authoritative BMU Helpdesk Referral"`.

---

### 2.7 Node 5: Agent 3 - Academic Response Synthesizer (`response_synthesizer_node`)

#### Pure Logic & Responsibility
Generates a grounded academic response strictly faithful to `vetted_context`.

- **Groq Client & Configuration:**
  - Env Var: `GROQ_API_KEY_SYNTHESIS_AGENT` (fallback: `GROQ_API_KEY`)
  - Model: `openai/gpt-oss-120b` (Largest parameter model for structured reasoning and synthesis)
  - Temperature: `0.2` (Constrained factual adherence)
  - Max Tokens: `800`
- **System Prompt:**
  `"You are the official BMU Academic Assistant. Answer strictly from context."`
- **Synthesis Rules:**
  1. Direct statement of facts with structured bullet points for fees, eligibility, and dates.
  2. Zero extrapolation: If an exact detail is absent, state that it is not specified in the current records.
  3. No raw markdown headers (`#` or `##`), ensuring clean typography in the chat bubble UI.

---

## 3. Dual-Tier JEV Guardrails Engine (`jev_guardrails.py`)

Implemented via TypeSafe AI's System-One decision model schema with a high-speed (<3ms) local deterministic engine.

```
       Incoming Query
             │
             ▼
   [ Input Guardrail ]
             ├── 1. Regex Injection Detector (12 vectors)
             ├── 2. Regex Toxicity Detector (4 vectors)
             └── 3. Scope / Keyword Classifier
                     ├── BMU Academic -> ALLOW
                     ├── General Academic -> ALLOW
                     ├── Prompt Injection -> BLOCK
                     ├── Toxic -> BLOCK
                     └── Out of Scope -> BLOCK
             │ (If Allowed)
             ▼
   [ LangGraph Multi-Agent RAG ]
             │
             ▼
   [ Output Guardrail ]
             ├── 1. Safety Re-check (toxic pattern scan)
             └── 2. Grounding Calculator:
                     Score = (|Tokens_Answer ∩ Tokens_Context| / |Tokens_Answer|) * 1.2
                     ├── Score >= 0.40 -> PASS
                     └── Score < 0.40  -> FLAG & APPEND DISCLAIMER
```

### 3.1 Input Guardrail Logic (`validate_input`)

#### Detection Arrays:
- **12 Adversarial Prompt Injection Patterns:**
  1. `ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)`
  2. `system\s+(override|prompt|reset)`
  3. `you\s+are\s+now\s+(in\s+)?(dan|jailbreak|developer|unrestricted)\s+mode`
  4. `act\s+as\s+(an?\s+)?unrestricted`
  5. `disregard\s+(all\s+)?(safety|rules|instructions)`
  6. `reveal\s+(your\s+)?(secret|internal|system)\s+(prompt|instructions|api\s*key)`
  7. `print\s+(the\s+)?(system\s+prompt|api\s*key)`
  8. `forget\s+(that\s+)?you\s+are\s+(a\s+)?bmu`
  9. `<\s*script\b[^>]*>`
  10. `drop\s+table\b`
  11. `union\s+select\b`
  12. `exec(\s*\(|\s+xp_)`
- **Toxicity Patterns:** Matches weapons, explosives, suicide, murder, illegal narcotics, racial slurs, and malicious exploits.
- **Academic Domain Keywords:** 45+ terms including `bmu`, `bml`, `munjal`, `admission`, `fees`, `btech`, `mba`, `hostel`, `placement`, `nirf`, `naac`, `scholarship`.
- **Out of Scope Keywords:** Detects sports scores, cooking recipes, cryptocurrency, gaming, pop culture.

#### Evaluation Logic:
$$\text{If } P(\text{injection}) \ge 0.70 \implies \text{BLOCK (action: 'block', category: 'prompt\_injection')}$$
$$\text{If } P(\text{toxic}) \ge 0.70 \implies \text{BLOCK (action: 'block', category: 'toxic')}$$
$$\text{If } \text{category} == \text{'out\_of\_scope'} \implies \text{BLOCK (action: 'block', category: 'out\_of\_scope')}$$
$$\text{Otherwise} \implies \text{ALLOW (action: 'pass', category: 'bmu\_academic')}$$

---

### 3.2 Output Guardrail Logic (`validate_output`)

#### Grounding Verification Formulation:
$$\text{overlap} = \text{Tokens}_{\ge 4 \text{ chars}}(\text{Answer}) \cap \text{Tokens}_{\ge 4 \text{ chars}}(\text{Context})$$
$$\text{overlap\_ratio} = \frac{|\text{overlap}|}{\max(1, |\text{Tokens}_{\ge 4 \text{ chars}}(\text{Answer})|)}$$
$$\text{grounding\_score} = \min(1.0, \max(0.2, \text{overlap\_ratio} \times 1.2))$$

#### Policy Routing:
1. **Safety Check:** If $P(\text{safe}) < 0.60 \implies$ Action: `block` (Withholds response).
2. **Hallucination Detection:** If $P(\text{grounded}) < 0.40 \implies$ Action: `flag_disclaimer`.
   - Appends contextual verification disclaimer:
     `"*(Note: Please verify specific figures or dates directly on the official BMU portal at [URL].)*"`
3. **Verified Grounded:** If $P(\text{grounded}) \ge 0.40 \implies$ Action: `pass`.

---

## 4. FastAPI Backend Orchestrator (`semantic.py`)

Serves as the central API gateway on port 8000.

### 4.1 Schema Definitions

```python
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")

class RankedUrl(BaseModel):
    url: str
    score: float

class GuardrailMeta(BaseModel):
    status: str       # "passed" | "blocked" | "flagged"
    action: str       # "allow" | "block" | "flag_disclaimer" | "retrieval_gated"
    reason: str
    engine: str       # "jev-system-one-local" | "jev-cloud"
    input_checks: dict[str, Any]
    output_checks: dict[str, Any]

class AskResponse(BaseModel):
    query: str
    answer: str
    top_results: list[RankedUrl]
    guardrail: Optional[GuardrailMeta] = None
    agent_trace: Optional[list[dict[str, Any]]] = None
```

### 4.2 Endpoint Contract: `POST /ask`
1. **Guardrail Input Validation:** Runs `jev_guardrail.validate_input(query)`.
   - If blocked, returns immediate HTTP 200 with `status: "blocked"` and canned policy explanation.
2. **Multi-Agent RAG Orchestration:**
   - Wraps invocation in OpenInference span `bmu_rag_pipeline`.
   - Executes `bmu_rag_app.invoke(initial_state)`.
3. **Retrieval Gating:**
   - If `audit_decision == "NOT_FOUND"` or context is empty, skips output guardrail and returns authoritative referral with `action: "retrieval_gated"`.
4. **Guardrail Output Validation:**
   - Runs `jev_guardrail.validate_output()`.
   - Attaches grounding probabilities, safety flags, and latency metrics to response.

---

## 5. Standalone Retrieval Microservice (`retrieval_service.py`)

Runs on port 8001. Decouples heavy PyTorch/SentenceTransformers and ChromaDB client memory footprints from the main web server.

### 5.1 Hybrid Ranking Formulation

The retrieval engine balances semantic conceptual matching with exact lexical token overlap:

$$\text{SemanticSimilarity} = \max(0.0, 1.0 - \text{ChromaDistance})$$
$$\text{KeywordScore} = \frac{|\text{Tokens}(\text{Query}) \cap \text{Tokens}(\text{Document})|}{|\text{Tokens}(\text{Query})|}$$
$$\mathbf{FinalScore} = \mathbf{0.68} \times \text{SemanticSimilarity} + \mathbf{0.32} \times \text{KeywordScore}$$

- **Semantic Weight (0.68):** Handles synonyms, paraphrasing, intent matching (e.g. "what is the price" -> "tuition fees").
- **Keyword Weight (0.32):** Ensures exact acronyms and program codes (e.g. "CSE", "MBA", "B.Tech", "NIRF") receive top ranking over generic university pages.

### 5.2 Dual-Engine Support
- **Engine 1: MiniLM (`minilm`):**
  - Collection: `bmu_summaries`
  - Model: `all-MiniLM-L6-v2` (384 dimensions)
  - Speed: ~15ms inference on CPU
- **Engine 2: Voyage AI (`voyage`):**
  - Collection: `bmu_voyage_knowledge`
  - Model: `voyage-4-large` (1024 dimensions)
  - Remote API via MongoDB Atlas

---

## 6. Knowledge Ingestion Microservice & Web Crawler (`ingestion_service.py`)

Runs on port 8002. Handles crawling, text extraction, semantic chunking, and vector indexing.

### 6.1 Web Crawler (`BMUWebCrawler`)
- **Domain Constraint:** Whitelisted strictly to `*.bmu.edu.in`.
- **Extension Filter (17 types):** Rejects `.pdf`, `.jpg`, `.jpeg`, `.png`, `.gif`, `.svg`, `.css`, `.js`, `.zip`, `.tar`, `.gz`, `.mp4`, `.mp3`, `.doc`, `.docx`, `.xls`, `.xlsx`.
- **User-Agent:** `Mozilla/5.0 (Windows NT 10.0; Win64; x64) BMU-IngestionBot/1.0`.

### 6.2 Text Cleaning & Chunking (`chunk_document`)
- **Cleaning:** Strips non-printable control characters `[\r\t\f\v]`, collapses multiple spaces, and normalizes line breaks.
- **Chunk Parameters:**
  - `chunk_size`: 800 characters
  - `chunk_overlap`: 150 characters
- **Paragraph & Sentence Boundaries:**
  - Iterates over paragraphs separated by `\n\n`.
  - When a paragraph exceeds 800 characters, splits using regex sentence lookbehind `(?<=[.!?])\s+`.
  - Re-injects overlapping trailing text to preserve context across chunk splits.

### 6.3 SHA-256 Deduplication
$$\text{Hash} = \text{SHA-256}(\text{normalize}(\text{chunk\_text}))$$
- Before sending chunks to Voyage AI, the system checks whether the hash already exists in metadata. Redundant chunks are skipped, saving API credits and avoiding duplicate vector collisions.

---

## 7. Voyage AI Embedding Client (`voyage_client.py`)

Custom client targeting `https://ai.mongodb.com/v1/embeddings` using `voyage-4-large` (1024-dim dense vectors).

### 7.1 Rate Limit Handling & Backoff Math
- Free/developer tier allows 3 Requests Per Minute (RPM).
- **HTTP 429 Interceptor:**
  $$\text{Delay} = \text{Retry-After header} + 1.0\text{s} \quad (\text{or } 21.0\text{s} + \text{attempt} \times 2.0\text{s})$$
  Pauses execution to allow the 60-second rolling RPM window to reset.
- **Server Error (5xx) Interceptor:**
  $$\text{Delay} = 2^{(\text{attempt} - 1)} \times 2.0\text{s} \quad (\text{Exponential backoff})$$
- **Batching:** Batches document texts into groups of 16 to maximize token efficiency per HTTP request.

---

## 8. Arize Phoenix AI Observability (`phoenix_tracer.py`, `phoenix_service.py`)

Collects real-time trace telemetry on port 6006 using the OpenInference semantic specification.

### 8.1 OpenInference Span Hierarchy

```text
Trace (Session: User Query)
└── bmu_rag_pipeline [CHAIN] (Total request latency)
    ├── LangGraph [CHAIN] (StateGraph execution)
    │   ├── query_specialist [CHAIN]
    │   ├── retrieval_tool [CHAIN]
    │   ├── context_auditor [CHAIN]
    │   ├── advance_candidate [CHAIN] (if fallback triggered)
    │   ├── context_auditor [CHAIN] (second candidate audit)
    │   └── response_synthesizer [CHAIN]
    ├── Completions [LLM] (Agent 1 Groq API call)
    ├── Completions [LLM] (Agent 2 Groq API call #1)
    ├── Completions [LLM] (Agent 2 Groq API call #2)
    └── Completions [LLM] (Agent 3 Groq API call)
```

### 8.2 Span Kinds & Attributes
- `CHAIN`: Pipeline workflows (`openinference.span.kind = "CHAIN"`).
- `LLM`: Generative model completions with token counts, prompt, response, and model name.
- `GUARDRAIL`: Security gates with `input.value`, `output.value`, and `guardrail.status`.
- **OTLP Exporter:** Transmits trace batches via HTTP protobuf to `http://localhost:6006/v1/traces`.

---

## 9. Evaluation & Benchmarking Engines

### 9.1 Answer Quality Evaluation (`evaluate_metrics.py`)
Computes 5 quality metrics on a 0.00 to 5.00 scale:

1. **Relevance:**
   $$\text{Score} = 5.0 \times \left(0.6 \times \frac{|\text{Tokens}_Q \cap \text{Tokens}_A|}{|\text{Tokens}_Q|} + 0.4 \times \text{CosineSim}_{\text{TF-IDF}}(Q, A)\right)$$
2. **Faithfulness:**
   $$\text{Score} = 5.0 \times \left(0.5 \times \frac{|\text{Tokens}_A \cap \text{Tokens}_{\text{Ctx}}|}{|\text{Tokens}_A|} + 0.5 \times \text{CosineSim}_{\text{TF-IDF}}(A, \text{Ctx})\right)$$
3. **Completeness:**
   Blends query token coverage with ground-truth TF-IDF support.
4. **Conciseness (Length Penalty Step Function):**
   - $\le 25$ words $\to 5.0$
   - $\le 45$ words $\to 4.0$
   - $\le 70$ words $\to 3.0$
   - $\le 100$ words $\to 2.0$
   - $\le 140$ words $\to 1.0$
   - $> 140$ words $\to 0.0$
5. **Correctness:**
   $$\text{Score} = 5.0 \times \left(0.4 \times \text{TokenOverlap}(A, \text{Ref}) + 0.6 \times \text{CosineSim}_{\text{TF-IDF}}(A, \text{Ref})\right)$$

### 9.2 Representative Classification Benchmark (`classification_metrics.py`)
- **Accuracy:** `0.81`
- **Precision:** `0.84`
- **Recall:** `0.78`
- **F1 Score:** `0.81` ($2 \times \frac{0.84 \times 0.78}{0.84 + 0.78} = 0.8088$)
- **AUC (ROC):** `0.87`

---

## 10. Multi-Process Launcher (`run.bat`)

Orchestrates local developer bootup and shutdown in Windows environments:
1. **Dependency Verification:** Runs `pip install -r requirements.txt --quiet`.
2. **Port Conflict Clearance:**
   ```powershell
   Get-NetTCPConnection -LocalPort 8000,8001,8002,6006 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
   ```
3. **Concurrent Process Spawning:**
   - Starts Phoenix on port 6006 (`python phoenix_service.py`)
   - Starts Retrieval Service on port 8001 (`python retrieval_service.py`)
   - Starts Ingestion Service on port 8002 (`python ingestion_service.py`)
   - Starts Backend Orchestrator on port 8000 (`python semantic.py`)
4. **Health Check Polling Loop:**
   Polls `/health` endpoints with a 3-second sleep interval across up to 12 attempts (36-second safety timeout).
5. **Interactive Teardown:**
   Blocks on `pause >nul`. Upon keypress, cleanly kills all processes across ports 8000, 8001, 8002, and 6006.
