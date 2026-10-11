import json
from pathlib import Path
import re
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import requests
import trafilatura

from evaluate_metrics import evaluate_answer
from semantic import _get_resources, _rank_results


def generate_queries(limit: int = 100) -> list[str]:
    topics = [
        "admissions",
        "eligibility",
        "application process",
        "fees",
        "scholarships",
        "hostel",
        "transport",
        "placements",
        "internships",
        "courses",
        "departments",
        "faculty",
        "campus facilities",
        "library",
        "sports",
        "events",
        "academic calendar",
        "exam rules",
        "attendance",
        "contact details",
    ]

    templates = [
        "What information is available about {topic} at BMU?",
        "Can you summarize BMU details regarding {topic}?",
        "Where can I find official BMU information about {topic}?",
        "Please explain BMU policies related to {topic}.",
        "What should students know about BMU {topic}?",
    ]

    queries: list[str] = []
    for topic in topics:
        for template in templates:
            queries.append(template.format(topic=topic))
            if len(queries) == limit:
                return queries
    return queries[:limit]


def fetch_text(url: str) -> str:
    if not url:
        return ""
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    # Protect against rare long-running extraction calls.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(trafilatura.extract, response.text)
        try:
            extracted = future.result(timeout=10)
        except FutureTimeoutError:
            return ""
    return extracted or ""

def build_answer_from_context(context: str, max_words: int = 55) -> str:
    text = re.sub(r"\s+", " ", (context or "").strip())
    if not text:
        return "Not found in context"

    # Use the first few sentences from extracted content as a concise answer.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    draft = " ".join(sentences[:2]).strip() or text

    words = draft.split()
    if len(words) > max_words:
        draft = " ".join(words[:max_words]).strip() + "..."
    return draft


def main() -> None:
    output_dir = Path("evaluation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "evaluation_100_results_partial.json"

    queries = generate_queries(100)
    results = []

    if partial_path.exists():
        try:
            results = json.loads(partial_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            results = []

    done_ids = {int(item.get("id", -1)) for item in results if item.get("status") == "ok"}

    collection, embedding_model, _ = _get_resources()

    for index, query in enumerate(queries, start=1):
        if index in done_ids:
            print(f"[{index}/100] skipped")
            continue

        try:
            query_embedding = embedding_model.encode([query.strip().lower()]).tolist()
            chroma_result = collection.query(query_embeddings=query_embedding, n_results=10)

            documents = chroma_result.get("documents", [[]])[0]
            metadatas = chroma_result.get("metadatas", [[]])[0]
            distances = chroma_result.get("distances", [[]])[0]

            ranked = _rank_results(query, documents, metadatas, distances, top_k=1)
            top_url = ranked[0]["url"] if ranked else ""

            ground_truth = fetch_text(top_url)
            context = ground_truth[:4000]
            answer = build_answer_from_context(context)

            metrics = evaluate_answer(
                context=context,
                question=query,
                answer=answer,
                ground_truth=ground_truth,
            )

            row = {
                "id": index,
                "query": query,
                "answer": answer,
                "url": top_url,
                "relevance": metrics.relevance,
                "faithfulness": metrics.faithfulness,
                "completeness": metrics.completeness,
                "conciseness": metrics.conciseness,
                "correctness": metrics.correctness,
                "final_score": metrics.final_score,
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "id": index,
                "query": query,
                "answer": "",
                "url": "",
                "relevance": 0.0,
                "faithfulness": 0.0,
                "completeness": 0.0,
                "conciseness": 0.0,
                "correctness": 0.0,
                "final_score": 0.0,
                "status": f"error: {exc}",
            }

        # Replace existing row with same id if present, otherwise append.
        replaced = False
        for pos, existing in enumerate(results):
            if int(existing.get("id", -1)) == index:
                results[pos] = row
                replaced = True
                break
        if not replaced:
            results.append(row)

        partial_path.write_text(
            json.dumps(sorted(results, key=lambda x: x.get("id", 0)), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        print(f"[{index}/100] {row['status']}")

    results = sorted(results, key=lambda x: x.get("id", 0))
    completed = [r for r in results if r["status"] == "ok"]

    summary = {
        "total_queries": len(results),
        "successful": len(completed),
        "failed": len(results) - len(completed),
        "avg_relevance": round(mean([r["relevance"] for r in completed]), 2) if completed else 0.0,
        "avg_faithfulness": round(mean([r["faithfulness"] for r in completed]), 2) if completed else 0.0,
        "avg_completeness": round(mean([r["completeness"] for r in completed]), 2) if completed else 0.0,
        "avg_conciseness": round(mean([r["conciseness"] for r in completed]), 2) if completed else 0.0,
        "avg_correctness": round(mean([r["correctness"] for r in completed]), 2) if completed else 0.0,
        "avg_final_score": round(mean([r["final_score"] for r in completed]), 2) if completed else 0.0,
    }

    (output_dir / "evaluation_100_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    (output_dir / "evaluation_100_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
