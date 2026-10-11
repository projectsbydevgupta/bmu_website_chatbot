import json
from dataclasses import dataclass, asdict


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    roc_points: list[tuple[float, float]]


# These are representative benchmark values (not computed from live predictions).
METRICS_VALUES = ClassificationMetrics(
    accuracy=0.81,
    precision=0.84,
    recall=0.78,
    f1_score=0.81,
    auc=0.87,
    roc_points=[
        (0.00, 0.00),
        (0.05, 0.28),
        (0.10, 0.49),
        (0.18, 0.66),
        (0.30, 0.80),
        (0.45, 0.89),
        (1.00, 1.00),
    ],
)


JUSTIFICATION = {
    "accuracy": (
        "Set to 0.81 to represent good overall correctness while acknowledging"
        " there is room for improvement in retrieval and answer grounding."
    ),
    "precision": (
        "Set to 0.84 to emphasize that when the system predicts a positive/"
        "relevant outcome, it is usually correct."
    ),
    "recall": (
        "Set to 0.78 to reflect that the model still misses some relevant"
        " cases, matching a realistic chatbot behavior."
    ),
    "f1_score": (
        "Set to 0.81 because it balances precision and recall. It is also"
        " consistent with 2*(P*R)/(P+R) for P=0.84 and R=0.78 (~0.809)."
    ),
    "auc": (
        "Set to 0.87 to represent strong ranking/discrimination ability across"
        " decision thresholds."
    ),
    "roc_points": (
        "ROC points are chosen to form a smooth upward curve typical of a"
        " reasonably strong classifier; this supports the AUC value used above."
    ),
    "note": (
        "These values are intentionally fixed for documentation, viva, or"
        " UI demonstration. For scientific reporting, replace with values"
        " computed from actual predictions and labels."
    ),
}


def print_table(metrics: ClassificationMetrics) -> None:
    rows = [
        ("Accuracy", metrics.accuracy),
        ("Precision", metrics.precision),
        ("Recall", metrics.recall),
        ("F1 Score", metrics.f1_score),
        ("AUC", metrics.auc),
    ]

    width_metric = max(len(name) for name, _ in rows)
    width_value = 6

    border = f"+-{'-' * width_metric}-+-{'-' * width_value}-+"
    print(border)
    print(f"| {'Metric'.ljust(width_metric)} | {'Value'.rjust(width_value)} |")
    print(border)
    for name, value in rows:
        print(f"| {name.ljust(width_metric)} | {value:>{width_value}.2f} |")
    print(border)


if __name__ == "__main__":
    payload = {
        "metrics": asdict(METRICS_VALUES),
        "justification": JUSTIFICATION,
    }

    print("=== Classification Metrics (JSON) ===")
    print(json.dumps(payload, indent=2, ensure_ascii=True))

    print("\n=== Metrics Table ===")
    print_table(METRICS_VALUES)
