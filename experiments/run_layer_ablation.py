"""
Run the layer aggregation ablation on labelled training data only.

The script extracts Qwen hidden states once, builds features for every
AGGREGATION_MODE, evaluates each mode with the repository probe pipeline, and
writes a compact comparison table to results/layer_ablation.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aggregation
from evaluate import run_evaluation
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data


DATA_FILE = ROOT / "data" / "dataset.csv"
OUTPUT_FILE = ROOT / "results" / "layer_ablation.csv"
BATCH_SIZE = 4
MODES = list(aggregation.SUPPORTED_AGGREGATION_MODES)


def _nanmean(values: list[float]) -> float:
    valid = [value for value in values if not np.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_features_for_all_modes(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    features_by_mode: dict[str, list[torch.Tensor]] = {mode: [] for mode in MODES}

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Extracting hidden states",
        unit="batch",
    ):
        batch_texts = texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            for mode in MODES:
                aggregation.AGGREGATION_MODE = mode
                feat = aggregation.aggregation_and_feature_extraction(
                    hidden[i],
                    mask[i],
                    use_geometric=False,
                )
                features_by_mode[mode].append(feat.cpu())

    return features_by_mode


def summarize(mode: str, fold_results: list[dict], feature_dim: int) -> dict:
    return {
        "mode": mode,
        "feature_dim": feature_dim,
        "val_accuracy": _nanmean(
            [result.get("val_accuracy", float("nan")) for result in fold_results]
        ),
        "val_f1": _nanmean(
            [result.get("val_f1", float("nan")) for result in fold_results]
        ),
        "val_auroc": _nanmean(
            [result.get("val_auroc", float("nan")) for result in fold_results]
        ),
        "test_accuracy": _nanmean(
            [result["test_accuracy"] for result in fold_results]
        ),
        "test_f1": _nanmean([result["test_f1"] for result in fold_results]),
        "test_auroc": _nanmean([result["test_auroc"] for result in fold_results]),
    }


def main() -> None:
    device = _device()
    print(f"Device     : {device}")
    print(f"Data       : {DATA_FILE}")
    print(f"Modes      : {', '.join(MODES)}")

    df = pd.read_csv(DATA_FILE)
    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    y = np.array([int(float(label)) for label in df["label"]])
    splits = split_data(y, df)

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    t0 = time.time()
    features_by_mode = extract_features_for_all_modes(texts, model, tokenizer, device)
    extract_time = time.time() - t0
    print(f"Feature extraction done in {extract_time:.1f} s")

    rows = []
    for mode in MODES:
        X = np.vstack([feature.numpy() for feature in features_by_mode[mode]])
        print(f"\nMode: {mode}  feature_dim={X.shape[1]}")
        fold_results = run_evaluation(splits, X, y, HallucinationProbe)
        rows.append(summarize(mode, fold_results, X.shape[1]))

    results = pd.DataFrame(rows).sort_values(
        by=["val_accuracy", "test_accuracy"],
        ascending=False,
        na_position="last",
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nLayer aggregation ablation")
    print(
        results[
            [
                "mode",
                "feature_dim",
                "val_accuracy",
                "val_f1",
                "val_auroc",
                "test_accuracy",
                "test_f1",
                "test_auroc",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
