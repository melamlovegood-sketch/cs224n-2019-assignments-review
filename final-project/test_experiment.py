"""Fast correctness checks for metric and threshold logic."""

from __future__ import annotations

from datasets import Dataset

from experiment import (
    compute_metrics,
    exact_match_score,
    f1_score,
    normalize_answer,
    predictions_at_threshold,
    tune_threshold,
)


def main() -> None:
    assert normalize_answer("The, Quick fox!") == "quick fox"
    assert exact_match_score("an answer", "Answer") == 1.0
    assert abs(f1_score("red blue", "red green") - 0.5) < 1e-9

    examples = Dataset.from_list(
        [
            {"id": "a", "answers": {"text": ["Paris"], "answer_start": [0]}},
            {"id": "b", "answers": {"text": [], "answer_start": []}},
        ]
    )
    records = {
        "a": {"best_non_null": "Paris", "score_diff": -2.0},
        "b": {"best_non_null": "London", "score_diff": 3.0},
    }
    predictions = predictions_at_threshold(records, 0.0)
    assert predictions == {"a": "Paris", "b": ""}
    metrics = compute_metrics(examples, predictions)
    assert metrics["exact_match"] == 100.0
    assert metrics["f1"] == 100.0
    threshold, calibrated = tune_threshold(examples, records)
    assert -2.0 <= threshold < 3.0
    assert calibrated["f1"] == 100.0
    print("All project unit tests passed.")


if __name__ == "__main__":
    main()
