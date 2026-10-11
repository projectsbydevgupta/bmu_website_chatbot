import argparse
import json
import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import requests
import trafilatura
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvaluationResult:
    relevance: float
    faithfulness: float
    completeness: float
    conciseness: float
    correctness: float
    final_score: float


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def ratio_to_five(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return round(value * 5.0, 2)


def safe_cosine_similarity(text_a: str, text_b: str) -> float:
    a = normalize_text(text_a)
    b = normalize_text(text_b)
    if not a or not b:
        return 0.0

    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform([a, b])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(max(0.0, min(1.0, score)))


def get_ground_truth_from_url(url: str, timeout: int = 15) -> str:
    if not url:
        return ""

    # Fast reachability check before extraction.
    try:
        requests.get(url, timeout=timeout)
    except requests.RequestException:
        return ""

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""

    clean_text = trafilatura.extract(downloaded)
    return normalize_text(clean_text or "")


def score_relevance(question: str, answer: str) -> float:
    q_tokens = tokenize(question)
    a_tokens = tokenize(answer)
    if not q_tokens:
        return 0.0

    overlap = len(q_tokens.intersection(a_tokens)) / len(q_tokens)
    semantic = safe_cosine_similarity(question, answer)
    return ratio_to_five((0.6 * overlap) + (0.4 * semantic))


def score_faithfulness(answer: str, context: str, ground_truth: str = "") -> float:
    source = normalize_text(context)
    if ground_truth:
        source = f"{source} {ground_truth}".strip()

    lexical = 0.0
    a_tokens = tokenize(answer)
    s_tokens = tokenize(source)
    if a_tokens:
        lexical = len(a_tokens.intersection(s_tokens)) / len(a_tokens)

    semantic = safe_cosine_similarity(answer, source)
    return ratio_to_five((0.5 * lexical) + (0.5 * semantic))


def score_completeness(question: str, answer: str, ground_truth: str = "") -> float:
    q_tokens = tokenize(question)
    a_tokens = tokenize(answer)
    if not q_tokens:
        return 0.0

    query_coverage = len(q_tokens.intersection(a_tokens)) / len(q_tokens)

    gt_support = 0.0
    if ground_truth:
        gt_support = safe_cosine_similarity(answer, ground_truth)

    blend = query_coverage if not ground_truth else (0.6 * query_coverage) + (0.4 * gt_support)
    return ratio_to_five(blend)


def score_conciseness(answer: str) -> float:
    words = re.findall(r"\S+", answer or "")
    count = len(words)

    if count == 0:
        return 0.0
    if count <= 25:
        return 5.0
    if count <= 45:
        return 4.0
    if count <= 70:
        return 3.0
    if count <= 100:
        return 2.0
    if count <= 140:
        return 1.0
    return 0.0


def score_correctness(answer: str, ground_truth: str = "", context: str = "") -> float:
    reference = normalize_text(ground_truth) or normalize_text(context)
    if not reference:
        return 0.0

    lexical = 0.0
    a_tokens = tokenize(answer)
    r_tokens = tokenize(reference)
    if a_tokens:
        lexical = len(a_tokens.intersection(r_tokens)) / len(a_tokens)

    semantic = safe_cosine_similarity(answer, reference)
    return ratio_to_five((0.4 * lexical) + (0.6 * semantic))


def evaluate_answer(
    context: str,
    question: str,
    answer: str,
    ground_truth: str = "",
) -> EvaluationResult:
    relevance = score_relevance(question, answer)
    faithfulness = score_faithfulness(answer, context, ground_truth)
    completeness = score_completeness(question, answer, ground_truth)
    conciseness = score_conciseness(answer)
    correctness = score_correctness(answer, ground_truth=ground_truth, context=context)

    final = round(mean([relevance, faithfulness, completeness, conciseness, correctness]), 2)

    return EvaluationResult(
        relevance=relevance,
        faithfulness=faithfulness,
        completeness=completeness,
        conciseness=conciseness,
        correctness=correctness,
        final_score=final,
    )


def as_table(result: EvaluationResult) -> str:
    rows = [
        ("Relevance", result.relevance),
        ("Faithfulness", result.faithfulness),
        ("Completeness", result.completeness),
        ("Conciseness", result.conciseness),
        ("Correctness", result.correctness),
        ("Final Score", result.final_score),
    ]

    label_width = max(len(r[0]) for r in rows)
    value_width = max(len(f"{r[1]:.2f}") for r in rows)

    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"
    lines = [border, f"| {'Metric'.ljust(label_width)} | {'Score'.rjust(value_width)} |", border]
    for metric, score in rows:
        lines.append(f"| {metric.ljust(label_width)} | {score:>{value_width}.2f} |")
    lines.append(border)
    return "\n".join(lines)


def load_text_arg(value: str) -> str:
    # If value points to an existing text file, read it; otherwise treat as raw text.
    try:
        with open(value, "r", encoding="utf-8") as file:
            return normalize_text(file.read())
    except OSError:
        return normalize_text(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chatbot answer quality (0-5 metrics).")
    parser.add_argument("--context", required=True, help="Context text or path to a context file")
    parser.add_argument("--question", required=True, help="Question text or path to a question file")
    parser.add_argument("--answer", required=True, help="Generated answer text or path to an answer file")
    parser.add_argument("--url", default="", help="Optional website URL for ground-truth extraction")
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "table"],
        help="Output format: json or table",
    )

    args = parser.parse_args()

    context = load_text_arg(args.context)
    question = load_text_arg(args.question)
    answer = load_text_arg(args.answer)

    ground_truth = get_ground_truth_from_url(args.url) if args.url else ""

    result = evaluate_answer(
        context=context,
        question=question,
        answer=answer,
        ground_truth=ground_truth,
    )

    if args.format == "json":
        output: dict[str, Any] = asdict(result)
        if args.url:
            output["ground_truth_used"] = bool(ground_truth)
        print(json.dumps(output, ensure_ascii=True, indent=2))
    else:
        print(as_table(result))


if __name__ == "__main__":
    main()
