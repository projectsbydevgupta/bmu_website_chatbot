Hybrid NLP Summarization Pipeline (Core of AI RAG Chatbot)
===============================================

🚀 Overview

This project implements a lightweight, practical summarization pipeline that converts large, noisy website content into compact, high-quality summaries for use in a Retrieval-Augmented Generation (RAG) system.

Why it exists
- University and institutional websites contain lots of repeated, noisy, or low-value text (menus, headers, footers, tables). Using raw pages in a RAG system increases latency, storage, and hurts retrieval quality.

Solution at a glance
- A hybrid extractive summarizer that combines semantic embeddings, centroid scoring, redundancy filtering, and light rule-based compression to produce short, factual summaries used only for retrieval (not final answer generation).

Simple pipeline diagram

  [Raw page] --> [Cleaning] --> [Sentence split]
       --> [Embed sentences]
       --> [Compute centroid]
       --> [Rank by similarity]
       --> [Select top diverse sentences]
       --> [Rule-based compression]
       --> [Summary stored in vector DB]

Key ideas (short)
- Use sentence embeddings (all-MiniLM-L6-v2) to capture semantic meaning.
- Rank sentences by cosine similarity to the document centroid (the "average meaning").
- Pick top relevant, diverse sentences and remove redundancy (sim > 0.75).
- Light cleanup rules: whitespace, stopword trimming, formatting normalization.

Step-by-step (friendly)
1. Data collection & cleaning
   - Scrape pages (Trafilatura recommended).
   - Remove HTML noise (navbars, repeated headers, menus).
2. Sentence segmentation
   - Split into clean sentences (regex-based) for reliable embeddings.
3. Sentence embeddings
   - Convert each sentence to a 384-d vector using MiniLM.
4. Relevance scoring
   - Compute document centroid (mean of sentence vectors).
   - Rank sentences by cosine similarity to centroid — closest represent the main ideas.
5. Redundancy removal
   - Skip sentences too similar to already chosen ones (threshold ~0.75).
6. Selection
   - Pick top 3 relevant + diverse sentences per document (configurable).
7. Rule-based compression
   - Trim filler, normalize formatting, remove extra whitespace.

Why this works
- Extractive: keeps factual correctness (no hallucinations).
- Embedding-based ranking: captures semantic importance, not just keywords.
- Small, dense summaries reduce vector DB size and speed up retrieval.

Results & impact (realistic summary)
- Compression: from ~40,000 lines to ~2,500 lines (~16x reduction).
- Retrieval quality improved (example: pipeline contributed to ~0.81 F1 on retrieval tasks).
- Lower latency, reduced storage and faster RAG responses.

Integration with RAG
- Summaries are stored in a vector DB (e.g., Chroma).
- Retrieval uses the compact summaries for fast candidate selection.
- The full document is re-fetched only when needed to generate the final LLM answer.

Practical tips
- Tune redundancy threshold for your corpus (0.7–0.8 is a good start).
- Adapt summary length to document size (longer pages → more sentences).
- Consider query-aware summarization as a future improvement.

Challenges
- Inconsistent site structure and noisy academic content (tables, lists).
- Choosing the right redundancy threshold to balance coverage vs duplication.

Future improvements
- Adaptive summary length based on document complexity.
- Query-aware or dynamic summarization for better relevance to user queries.
- Lightweight abstractive refinement for readability (optional).

License & contact
- This repo contains the core summarization code. Use and adapt as needed.

Small visual (ASCII) — pipeline in one line

  RAW → CLEAN → SENTENCES → EMBED → CENTROID → RANK → DEDUP → COMPRESS → SUMMARY

Happy summarizing! ✨
