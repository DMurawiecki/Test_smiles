"""
Run spectral geometry ablations on the best layer aggregation baseline.

The experiment fixes AGGREGATION_MODE="mean_21_24" and compares scalar spectral
features computed from the Gram spectrum of the last answer-token hidden states.
Splits are provided by splitting.py; this script does not alter that protocol.
"""

from __future__ import annotations

import json
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
from probe import HallucinationProbe, permutation_importance_by_group
from splitting import split_data


DATA_FILE = ROOT / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"
ABLATION_FILE = RESULTS_DIR / "spectral_ablation.csv"
IMPORTANCE_FILE = RESULTS_DIR / "spectral_feature_importance.csv"
FEATURE_NAMES_FILE = RESULTS_DIR / "feature_names.json"
BATCH_SIZE = 4
AGGREGATION_MODE = "mean_21_24"
SPECTRAL_MODES = (
    "none",
    "top_eigenvalues",
    "sum_eigenvalues",
    "logdet",
    "effective_rank",
    "participation_ratio",
    "condition_number",
    "spectral_entropy",
    "all_without_condition_number",
    "all",
)


def _nanmean(values: list[float]) -> float:
    valid = [value for value in values if not np.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _summarize(
    spectral_mode: str,
    fold_results: list[dict],
    feature_dim: int,
) -> dict:
    return {
        "aggregation_mode": AGGREGATION_MODE,
        "spectral_mode": spectral_mode,
        "feature_dim": feature_dim,
        "n_spectral_features": aggregation.count_spectral_features(spectral_mode),
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


def extract_features_for_all_spectral_modes(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[str]]]:
    features_by_mode: dict[str, list[torch.Tensor]] = {
        mode: [] for mode in SPECTRAL_MODES
    }
    feature_names_by_mode: dict[str, list[str]] = {}

    aggregation.AGGREGATION_MODE = AGGREGATION_MODE
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
            for spectral_mode in SPECTRAL_MODES:
                aggregation.SPECTRAL_MODE = spectral_mode
                feat = aggregation.aggregation_and_feature_extraction(
                    hidden[i],
                    mask[i],
                    use_geometric=False,
                )
                features_by_mode[spectral_mode].append(feat.cpu())
                if spectral_mode not in feature_names_by_mode:
                    feature_names_by_mode[spectral_mode] = (
                        aggregation.get_last_feature_names()
                    )

    return features_by_mode, feature_names_by_mode


def compute_validation_importance(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
    feature_names: list[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    fold_frames: list[pd.DataFrame] = []
    for fold_idx, (idx_train, idx_val, _) in enumerate(splits):
        if idx_val is None:
            continue
        probe = HallucinationProbe()
        probe.fit(X[idx_train], y[idx_train])
        probe.fit_hyperparameters(X[idx_val], y[idx_val])
        frame = permutation_importance_by_group(
            probe,
            X[idx_val],
            y[idx_val],
            feature_names,
            n_repeats=n_repeats,
            random_state=42 + fold_idx,
        )
        if not frame.empty:
            frame["fold"] = fold_idx + 1
            fold_frames.append(frame)

    if not fold_frames:
        return pd.DataFrame(
            columns=[
                "group",
                "metric",
                "baseline_score",
                "permuted_score_mean",
                "permuted_score_std",
                "importance_mean",
                "importance_std",
                "n_features",
            ]
        )

    raw = pd.concat(fold_frames, ignore_index=True)
    rows: list[dict] = []
    for (group, metric), part in raw.groupby(["group", "metric"], sort=False):
        rows.append(
            {
                "group": group,
                "metric": metric,
                "baseline_score": part["baseline_score"].mean(),
                "permuted_score_mean": part["permuted_score_mean"].mean(),
                "permuted_score_std": part["permuted_score_std"].mean(),
                "importance_mean": part["importance_mean"].mean(),
                "importance_std": part["importance_mean"].std(ddof=0),
                "n_features": int(part["n_features"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["metric", "importance_mean"],
        ascending=[True, False],
    )


def print_comparison(results: pd.DataFrame) -> str:
    baseline = results.loc[results["spectral_mode"] == "none"].iloc[0]
    spectral = results.loc[results["spectral_mode"] != "none"]
    best_acc = spectral.sort_values(
        by=["val_accuracy", "val_auroc"],
        ascending=False,
    ).iloc[0]
    best_auroc = spectral.sort_values(
        by=["val_auroc", "val_accuracy"],
        ascending=False,
    ).iloc[0]

    print("\nBaseline SPECTRAL_MODE=none")
    print(
        baseline[
            [
                "val_accuracy",
                "val_auroc",
                "test_accuracy",
                "test_auroc",
            ]
        ].to_string()
    )

    print("\nBest spectral mode by val_accuracy")
    print(best_acc["spectral_mode"])
    print(
        f"delta_val_accuracy={best_acc['val_accuracy'] - baseline['val_accuracy']:.4f}  "
        f"delta_val_auroc={best_acc['val_auroc'] - baseline['val_auroc']:.4f}  "
        f"delta_test_accuracy={best_acc['test_accuracy'] - baseline['test_accuracy']:.4f}  "
        f"delta_test_auroc={best_acc['test_auroc'] - baseline['test_auroc']:.4f}"
    )

    print("\nBest spectral mode by val_auroc")
    print(best_auroc["spectral_mode"])
    print(
        f"delta_val_accuracy={best_auroc['val_accuracy'] - baseline['val_accuracy']:.4f}  "
        f"delta_val_auroc={best_auroc['val_auroc'] - baseline['val_auroc']:.4f}  "
        f"delta_test_accuracy={best_auroc['test_accuracy'] - baseline['test_accuracy']:.4f}  "
        f"delta_test_auroc={best_auroc['test_auroc'] - baseline['test_auroc']:.4f}"
    )
    return str(best_acc["spectral_mode"])


def main() -> None:
    device = _device()
    print(f"Device          : {device}")
    print(f"Data            : {DATA_FILE}")
    print(f"Aggregation mode: {AGGREGATION_MODE}")
    print(f"Spectral modes  : {', '.join(SPECTRAL_MODES)}")

    df = pd.read_csv(DATA_FILE)
    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    y = np.array([int(float(label)) for label in df["label"]])
    splits = split_data(y, df)

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    t0 = time.time()
    features_by_mode, feature_names_by_mode = extract_features_for_all_spectral_modes(
        texts,
        model,
        tokenizer,
        device,
    )
    print(f"Feature extraction done in {time.time() - t0:.1f} s")

    rows: list[dict] = []
    matrices: dict[str, np.ndarray] = {}
    for spectral_mode in SPECTRAL_MODES:
        aggregation.SPECTRAL_MODE = spectral_mode
        X = np.vstack([feature.numpy() for feature in features_by_mode[spectral_mode]])
        matrices[spectral_mode] = X
        feature_names_by_mode[spectral_mode] = aggregation.build_feature_names(
            X.shape[1]
        )
        print(
            f"\nSpectral mode: {spectral_mode}  "
            f"feature_dim={X.shape[1]}  "
            f"n_spectral={aggregation.count_spectral_features(spectral_mode)}"
        )
        fold_results = run_evaluation(splits, X, y, HallucinationProbe)
        rows.append(_summarize(spectral_mode, fold_results, X.shape[1]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with FEATURE_NAMES_FILE.open("w") as f:
        json.dump(feature_names_by_mode, f, indent=2)

    results = pd.DataFrame(rows).sort_values(
        by=["val_accuracy", "val_auroc", "test_accuracy"],
        ascending=False,
        na_position="last",
    )
    results.to_csv(ABLATION_FILE, index=False)

    print("\nSpectral ablation")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {ABLATION_FILE}")
    print(f"Saved: {FEATURE_NAMES_FILE}")

    best_mode = print_comparison(results)
    if best_mode == "none":
        spectral_candidates = results.loc[results["spectral_mode"] != "none"]
        best_mode = str(spectral_candidates.iloc[0]["spectral_mode"])

    print(f"\nComputing validation permutation importance for: {best_mode}")
    importance = compute_validation_importance(
        matrices[best_mode],
        y,
        splits,
        feature_names_by_mode[best_mode],
    )
    importance.to_csv(IMPORTANCE_FILE, index=False)
    print(f"Saved: {IMPORTANCE_FILE}")

    if not importance.empty:
        print("\nTop feature importance groups")
        print(
            importance.sort_values(
                by=["metric", "importance_mean"],
                ascending=[True, False],
            )
            .groupby("metric")
            .head(6)
            .to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )

    print(
        "\nInterpretation: this experiment checks whether spectral properties of "
        "the hidden-state cloud for the last answer tokens add hallucination "
        "signal beyond pooled mean(21-24) hidden vectors. Top eigenvalues track "
        "dominant variance directions; sum eigenvalues tracks total spread; "
        "logdet tracks cloud volume; effective rank and participation ratio "
        "track effective dimensionality; condition number tracks anisotropy; "
        "spectral entropy tracks spectrum uniformity."
    )


if __name__ == "__main__":
    main()
