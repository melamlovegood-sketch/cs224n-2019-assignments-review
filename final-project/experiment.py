"""Reproducible SQuAD 2.0 experiment for the CS224N default final project.

The experiment compares a SQuAD 1.1 DistilBERT checkpoint with the same model
fine-tuned on SQuAD 2.0.  It then calibrates the no-answer threshold on a
separate calibration split and reports final metrics on a held-out evaluation
split.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import platform
import random
import re
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)


MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"
DATASET_NAME = "rajpurkar/squad_v2"


@dataclass
class ExperimentConfig:
    model_name: str = MODEL_NAME
    train_size: int = 12_000
    validation_size: int = 3_000
    calibration_size: int = 1_500
    max_length: int = 256
    doc_stride: int = 64
    max_answer_length: int = 30
    n_best_size: int = 20
    train_batch_size: int = 12
    eval_batch_size: int = 24
    gradient_accumulation_steps: int = 2
    epochs: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 224
    output_dir: str = "project"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_train_features(examples: dict[str, list[Any]], tokenizer: Any, config: ExperimentConfig) -> dict[str, Any]:
    tokenized = tokenizer(
        [question.strip() for question in examples["question"]],
        examples["context"],
        max_length=config.max_length,
        truncation="only_second",
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    start_positions: list[int] = []
    end_positions: list[int] = []

    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]
        answers = examples["answers"][sample_index]

        if not answers["answer_start"]:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        token_start = 0
        while sequence_ids[token_start] != 1:
            token_start += 1
        token_end = len(input_ids) - 1
        while sequence_ids[token_end] != 1:
            token_end -= 1

        if offsets[token_start][0] > start_char or offsets[token_end][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while token_start < len(offsets) and offsets[token_start][0] <= start_char:
            token_start += 1
        start_positions.append(token_start - 1)
        while offsets[token_end][1] >= end_char:
            token_end -= 1
        end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_validation_features(examples: dict[str, list[Any]], tokenizer: Any, config: ExperimentConfig) -> dict[str, Any]:
    tokenized = tokenizer(
        [question.strip() for question in examples["question"]],
        examples["context"],
        max_length=config.max_length,
        truncation="only_second",
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    example_ids: list[str] = []

    for feature_index, sample_index in enumerate(sample_mapping):
        example_ids.append(examples["id"][sample_index])
        sequence_ids = tokenized.sequence_ids(feature_index)
        tokenized["offset_mapping"][feature_index] = [
            offset if sequence_ids[token_index] == 1 else None
            for token_index, offset in enumerate(tokenized["offset_mapping"][feature_index])
        ]

    tokenized["example_id"] = example_ids
    return tokenized


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(character for character in value if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not truth_tokens:
        return float(prediction_tokens == truth_tokens)
    common = collections.Counter(prediction_tokens) & collections.Counter(truth_tokens)
    shared = sum(common.values())
    if shared == 0:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_scores(prediction: str, answers: dict[str, list[Any]]) -> tuple[float, float]:
    gold_answers = [answer for answer in answers["text"] if normalize_answer(answer)] or [""]
    return (
        max(exact_match_score(prediction, answer) for answer in gold_answers),
        max(f1_score(prediction, answer) for answer in gold_answers),
    )


@torch.inference_mode()
def predict_logits(model: torch.nn.Module, features: Dataset, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    input_columns = [column for column in ("input_ids", "attention_mask", "token_type_ids") if column in features.column_names]
    model_inputs = features.remove_columns([column for column in features.column_names if column not in input_columns])
    model_inputs.set_format("torch")
    loader = DataLoader(model_inputs, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    all_start: list[np.ndarray] = []
    all_end: list[np.ndarray] = []
    model.eval()
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            output = model(**batch)
        all_start.append(output.start_logits.float().cpu().numpy())
        all_end.append(output.end_logits.float().cpu().numpy())
    return np.concatenate(all_start), np.concatenate(all_end)


def build_prediction_records(
    examples: Dataset,
    features: Dataset,
    raw_predictions: tuple[np.ndarray, np.ndarray],
    tokenizer: Any,
    config: ExperimentConfig,
) -> dict[str, dict[str, Any]]:
    start_logits, end_logits = raw_predictions
    example_to_features: dict[str, list[int]] = collections.defaultdict(list)
    for feature_index, example_id in enumerate(features["example_id"]):
        example_to_features[example_id].append(feature_index)

    records: dict[str, dict[str, Any]] = {}
    for example in tqdm(examples, desc="Post-processing", leave=False):
        best_answer = ""
        best_score = -math.inf
        null_score = math.inf
        for feature_index in example_to_features[example["id"]]:
            offsets = features[feature_index]["offset_mapping"]
            input_ids = features[feature_index]["input_ids"]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            null_score = min(null_score, float(start_logits[feature_index][cls_index] + end_logits[feature_index][cls_index]))
            start_indexes = np.argsort(start_logits[feature_index])[-config.n_best_size :][::-1]
            end_indexes = np.argsort(end_logits[feature_index])[-config.n_best_size :][::-1]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offsets)
                        or end_index >= len(offsets)
                        or offsets[start_index] is None
                        or offsets[end_index] is None
                        or end_index < start_index
                        or end_index - start_index + 1 > config.max_answer_length
                    ):
                        continue
                    score = float(start_logits[feature_index][start_index] + end_logits[feature_index][end_index])
                    if score > best_score:
                        start_char, end_char = offsets[start_index][0], offsets[end_index][1]
                        best_answer = example["context"][start_char:end_char]
                        best_score = score

        if best_score == -math.inf:
            best_score = null_score
        records[example["id"]] = {
            "best_non_null": best_answer,
            "score_diff": float(null_score - best_score),
        }
    return records


def predictions_at_threshold(records: dict[str, dict[str, Any]], threshold: float) -> dict[str, str]:
    return {
        example_id: "" if record["score_diff"] > threshold else record["best_non_null"]
        for example_id, record in records.items()
    }


def compute_metrics(examples: Iterable[dict[str, Any]], predictions: dict[str, str]) -> dict[str, float]:
    exact = 0.0
    f1 = 0.0
    answerable_exact = 0.0
    answerable_f1 = 0.0
    unanswerable_exact = 0.0
    answerable_count = 0
    unanswerable_count = 0
    examples_list = list(examples)
    for example in examples_list:
        prediction = predictions[example["id"]]
        em_value, f1_value = answer_scores(prediction, example["answers"])
        exact += em_value
        f1 += f1_value
        if example["answers"]["text"]:
            answerable_count += 1
            answerable_exact += em_value
            answerable_f1 += f1_value
        else:
            unanswerable_count += 1
            unanswerable_exact += em_value
    count = max(1, len(examples_list))
    return {
        "exact_match": 100.0 * exact / count,
        "f1": 100.0 * f1 / count,
        "answerable_exact_match": 100.0 * answerable_exact / max(1, answerable_count),
        "answerable_f1": 100.0 * answerable_f1 / max(1, answerable_count),
        "unanswerable_accuracy": 100.0 * unanswerable_exact / max(1, unanswerable_count),
        "count": len(examples_list),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
    }


def tune_threshold(examples: Dataset, records: dict[str, dict[str, Any]]) -> tuple[float, dict[str, float]]:
    scored: list[tuple[float, float, float, float, float]] = []
    for example in examples:
        record = records[example["id"]]
        non_null_em, non_null_f1 = answer_scores(record["best_non_null"], example["answers"])
        null_em, null_f1 = answer_scores("", example["answers"])
        scored.append((record["score_diff"], non_null_em, non_null_f1, null_em, null_f1))
    scored.sort(key=lambda item: item[0])
    exact_total = sum(item[3] for item in scored)
    f1_total = sum(item[4] for item in scored)
    best_threshold = scored[0][0] - 1e-6
    best_f1 = f1_total
    best_exact = exact_total
    index = 0
    while index < len(scored):
        threshold = scored[index][0]
        while index < len(scored) and scored[index][0] == threshold:
            _, non_em, non_f1, null_em, null_f1 = scored[index]
            exact_total += non_em - null_em
            f1_total += non_f1 - null_f1
            index += 1
        if f1_total > best_f1 or (f1_total == best_f1 and exact_total > best_exact):
            best_f1 = f1_total
            best_exact = exact_total
            best_threshold = threshold
    count = max(1, len(scored))
    metrics = compute_metrics(examples, predictions_at_threshold(records, best_threshold))
    metrics["calibration_objective_f1"] = 100.0 * best_f1 / count
    metrics["calibration_objective_exact_match"] = 100.0 * best_exact / count
    return float(best_threshold), metrics


def train_model(
    model: torch.nn.Module,
    train_features: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    columns = [column for column in ("input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions") if column in train_features.column_names]
    train_data = train_features.remove_columns([column for column in train_features.column_names if column not in columns])
    train_data.set_format("torch")
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train_data,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=default_data_collator,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
    total_steps = update_steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not use_bfloat16)
    history: list[dict[str, float]] = []
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epochs):
        progress = tqdm(loader, desc=f"Training epoch {epoch + 1}/{config.epochs}")
        running_loss = 0.0
        for batch_index, batch in enumerate(progress, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                output = model(**batch)
                loss = output.loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(loss.detach()) * config.gradient_accumulation_steps
            should_update = batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(loader)
            if should_update:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale() if scaler.is_enabled() else None
                scaler.step(optimizer)
                scaler.update()
                optimizer_ran = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
                if optimizer_ran:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                mean_loss = running_loss / batch_index
                history.append({
                    "step": global_step,
                    "epoch": epoch + batch_index / len(loader),
                    "loss": mean_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                })
                progress.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return history


def evaluate_model(
    label: str,
    model: torch.nn.Module,
    tokenizer: Any,
    calibration_examples: Dataset,
    calibration_features: Dataset,
    evaluation_examples: Dataset,
    evaluation_features: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    calibration_logits = predict_logits(model, calibration_features, config.eval_batch_size, device)
    calibration_records = build_prediction_records(calibration_examples, calibration_features, calibration_logits, tokenizer, config)
    threshold, calibration_metrics = tune_threshold(calibration_examples, calibration_records)

    evaluation_logits = predict_logits(model, evaluation_features, config.eval_batch_size, device)
    evaluation_records = build_prediction_records(evaluation_examples, evaluation_features, evaluation_logits, tokenizer, config)
    default_predictions = predictions_at_threshold(evaluation_records, 0.0)
    calibrated_predictions = predictions_at_threshold(evaluation_records, threshold)
    return {
        "label": label,
        "threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "default_metrics": compute_metrics(evaluation_examples, default_predictions),
        "calibrated_metrics": compute_metrics(evaluation_examples, calibrated_predictions),
        "records": evaluation_records,
        "default_predictions": default_predictions,
        "calibrated_predictions": calibrated_predictions,
    }


def write_error_analysis(
    path: Path,
    examples: Dataset,
    baseline: dict[str, Any],
    improved: dict[str, Any],
) -> None:
    entries: list[dict[str, Any]] = []
    for example in examples:
        example_id = example["id"]
        baseline_prediction = baseline["calibrated_predictions"][example_id]
        improved_prediction = improved["calibrated_predictions"][example_id]
        baseline_em, baseline_f1 = answer_scores(baseline_prediction, example["answers"])
        improved_em, improved_f1 = answer_scores(improved_prediction, example["answers"])
        if baseline_f1 == improved_f1:
            continue
        entries.append({
            "id": example_id,
            "title": example["title"],
            "question": example["question"],
            "context": example["context"],
            "gold_answers": example["answers"]["text"],
            "baseline_prediction": baseline_prediction,
            "improved_prediction": improved_prediction,
            "baseline_exact_match": baseline_em,
            "baseline_f1": baseline_f1,
            "improved_exact_match": improved_em,
            "improved_f1": improved_f1,
            "category": "improved" if improved_f1 > baseline_f1 else "regressed",
        })
    entries.sort(key=lambda item: (item["category"] != "improved", -(item["improved_f1"] - item["baseline_f1"])))
    selected = [entry for entry in entries if entry["category"] == "improved"][:12]
    selected += [entry for entry in entries if entry["category"] == "regressed"][:8]
    path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(output_dir: Path, history: list[dict[str, float]], results: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot([item["step"] for item in history], [item["loss"] for item in history], color="#2563eb")
    plt.xlabel("Optimizer step")
    plt.ylabel("Running training loss")
    plt.title("SQuAD 2.0 fine-tuning")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    labels = ["SQuAD 1.1\ncheckpoint", "SQuAD 2.0\nfine-tuned", "Fine-tuned +\ncalibration"]
    exact = [
        results["baseline"]["default_metrics"]["exact_match"],
        results["fine_tuned"]["default_metrics"]["exact_match"],
        results["fine_tuned"]["calibrated_metrics"]["exact_match"],
    ]
    f1 = [
        results["baseline"]["default_metrics"]["f1"],
        results["fine_tuned"]["default_metrics"]["f1"],
        results["fine_tuned"]["calibrated_metrics"]["f1"],
    ]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, exact, width, label="Exact Match", color="#0f766e")
    plt.bar(x + width / 2, f1, width, label="F1", color="#f59e0b")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.ylim(0, 100)
    plt.title("Held-out SQuAD 2.0 evaluation")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    for position, value in zip(x - width / 2, exact):
        plt.text(position, value + 1, f"{value:.1f}", ha="center", fontsize=9)
    for position, value in zip(x + width / 2, f1):
        plt.text(position, value + 1, f"{value:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison.png", dpi=180)
    plt.close()


def serializable_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items() if key not in {"records", "default_predictions", "calibrated_predictions"}}


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir).resolve()
    results_dir = output_dir / "results"
    model_dir = output_dir / "model"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset = load_dataset(DATASET_NAME)
    train_examples = dataset["train"].shuffle(seed=config.seed).select(range(min(config.train_size, len(dataset["train"]))))
    validation_examples = dataset["validation"].shuffle(seed=config.seed).select(range(min(config.validation_size, len(dataset["validation"]))))
    calibration_size = min(config.calibration_size, len(validation_examples) // 2)
    calibration_examples = validation_examples.select(range(calibration_size))
    evaluation_examples = validation_examples.select(range(calibration_size, len(validation_examples)))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    train_features = train_examples.map(
        lambda examples: prepare_train_features(examples, tokenizer, config),
        batched=True,
        remove_columns=train_examples.column_names,
        desc="Tokenizing training data",
    )
    calibration_features = calibration_examples.map(
        lambda examples: prepare_validation_features(examples, tokenizer, config),
        batched=True,
        remove_columns=calibration_examples.column_names,
        desc="Tokenizing calibration data",
    )
    evaluation_features = evaluation_examples.map(
        lambda examples: prepare_validation_features(examples, tokenizer, config),
        batched=True,
        remove_columns=evaluation_examples.column_names,
        desc="Tokenizing evaluation data",
    )

    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name).to(device)
    started = time.time()
    baseline = evaluate_model(
        "SQuAD 1.1 checkpoint",
        model,
        tokenizer,
        calibration_examples,
        calibration_features,
        evaluation_examples,
        evaluation_features,
        config,
        device,
    )
    baseline_seconds = time.time() - started

    started = time.time()
    history = train_model(model, train_features, config, device)
    training_seconds = time.time() - started
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    started = time.time()
    fine_tuned = evaluate_model(
        "SQuAD 2.0 fine-tuned",
        model,
        tokenizer,
        calibration_examples,
        calibration_features,
        evaluation_examples,
        evaluation_features,
        config,
        device,
    )
    fine_tuned_seconds = time.time() - started

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    report = {
        "config": asdict(config),
        "environment": environment,
        "dataset": {
            "name": DATASET_NAME,
            "train_examples": len(train_examples),
            "train_features": len(train_features),
            "calibration_examples": len(calibration_examples),
            "calibration_features": len(calibration_features),
            "evaluation_examples": len(evaluation_examples),
            "evaluation_features": len(evaluation_features),
        },
        "timing_seconds": {
            "baseline_evaluation": baseline_seconds,
            "training": training_seconds,
            "fine_tuned_evaluation": fine_tuned_seconds,
        },
        "baseline": serializable_evaluation(baseline),
        "fine_tuned": serializable_evaluation(fine_tuned),
    }
    (results_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (results_dir / "baseline_predictions.json").write_text(json.dumps(baseline["calibrated_predictions"], indent=2), encoding="utf-8")
    (results_dir / "fine_tuned_predictions.json").write_text(json.dumps(fine_tuned["calibrated_predictions"], indent=2), encoding="utf-8")
    write_error_analysis(results_dir / "error_analysis.json", evaluation_examples, baseline, fine_tuned)
    plot_results(results_dir, history, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--validation-size", type=int, default=3_000)
    parser.add_argument("--calibration-size", type=int, default=1_500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--doc-stride", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--output-dir", default="project")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end check.")
    args = parser.parse_args()
    if args.smoke:
        args.train_size = 32
        args.validation_size = 24
        args.calibration_size = 12
        args.train_batch_size = 4
        args.eval_batch_size = 8
        args.gradient_accumulation_steps = 1
        args.epochs = 1
        args.output_dir = "project_smoke"
    return ExperimentConfig(
        model_name=args.model_name,
        train_size=args.train_size,
        validation_size=args.validation_size,
        calibration_size=args.calibration_size,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_experiment(parse_args())
