"""
Sweep PCA dimensionality and probe classifier type.

Feature extraction is fixed to the current best baseline:
AGGREGATION_MODE="mean_21_24" and SPECTRAL_MODE="none".
The script caches that feature matrix once, then reuses it for every
PCA/classifier combination.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aggregation
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data


DATA_FILE = ROOT / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"
CACHE_FILE = RESULTS_DIR / "current_features_mean_21_24_spectral_none.npz"
OUTPUT_FILE = RESULTS_DIR / "pca_probe_ablation.csv"
SORTED_OUTPUT_FILE = RESULTS_DIR / "pca_probe_ablation_sorted.csv"

BATCH_SIZE = 4
AGGREGATION_MODE = "mean_21_24"
SPECTRAL_MODE = "none"
THRESHOLD_METRIC = "accuracy"

PCA_COMPONENTS = ("none", "16", "32", "64", "100", "128", "256")
PROBE_TYPES = (
    "logreg",
    "mlp",
    "logreg_l1",
    "svm_linear",
    "svm_rbf",
    "ridge",
    "knn",
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _nanmean(values: list[float]) -> float:
    valid = [value for value in values if not np.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def _metric_values(probe: HallucinationProbe, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = probe.predict(X)
    y_prob = probe.predict_proba(X)[:, 1]
    try:
        auroc = roc_auc_score(y, y_prob)
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auroc": auroc,
    }


def _run_probe_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
) -> dict:
    fold_rows: list[dict] = []
    thresholds: list[float] = []

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits):
        probe = HallucinationProbe()
        probe.fit(X[idx_train], y[idx_train])
        if idx_val is not None:
            probe.fit_hyperparameters(X[idx_val], y[idx_val])
        thresholds.append(probe.best_threshold)

        row: dict = {"fold": fold_idx + 1, "best_threshold": probe.best_threshold}
        if idx_val is not None:
            val = _metric_values(probe, X[idx_val], y[idx_val])
            row.update(
                {
                    "val_accuracy": val["accuracy"],
                    "val_f1": val["f1"],
                    "val_auroc": val["auroc"],
                }
            )

        test = _metric_values(probe, X[idx_test], y[idx_test])
        row.update(
            {
                "test_accuracy": test["accuracy"],
                "test_f1": test["f1"],
                "test_auroc": test["auroc"],
            }
        )
        fold_rows.append(row)

    return {
        "val_accuracy": _nanmean(
            [row.get("val_accuracy", float("nan")) for row in fold_rows]
        ),
        "val_f1": _nanmean([row.get("val_f1", float("nan")) for row in fold_rows]),
        "val_auroc": _nanmean(
            [row.get("val_auroc", float("nan")) for row in fold_rows]
        ),
        "test_accuracy": _nanmean([row["test_accuracy"] for row in fold_rows]),
        "test_f1": _nanmean([row["test_f1"] for row in fold_rows]),
        "test_auroc": _nanmean([row["test_auroc"] for row in fold_rows]),
        "best_threshold": float(np.mean(thresholds)) if thresholds else float("nan"),
    }


def _save_cache(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
) -> None:
    payload: dict[str, object] = {
        "X": X,
        "y": y,
        "ids": ids,
        "n_splits": np.asarray([len(splits)], dtype=np.int64),
        "metadata": np.asarray(
            [
                json.dumps(
                    {
                        "aggregation_mode": AGGREGATION_MODE,
                        "spectral_mode": SPECTRAL_MODE,
                        "feature_dim": int(X.shape[1]),
                    }
                )
            ]
        ),
    }
    for i, (idx_train, idx_val, idx_test) in enumerate(splits):
        payload[f"split_{i}_train"] = idx_train
        payload[f"split_{i}_val"] = (
            np.asarray([], dtype=np.int64) if idx_val is None else idx_val
        )
        payload[f"split_{i}_test"] = idx_test
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, **payload)


def _load_cache() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
] | None:
    if not CACHE_FILE.exists():
        return None

    data = np.load(CACHE_FILE, allow_pickle=False)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=int)
    ids = np.asarray(data["ids"], dtype=int)
    splits = []
    for i in range(int(data["n_splits"][0])):
        idx_train = np.asarray(data[f"split_{i}_train"], dtype=int)
        idx_val_raw = np.asarray(data[f"split_{i}_val"], dtype=int)
        idx_val = None if idx_val_raw.size == 0 else idx_val_raw
        idx_test = np.asarray(data[f"split_{i}_test"], dtype=int)
        splits.append((idx_train, idx_val, idx_test))
    return X, y, ids, splits


def _extract_current_features() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
]:
    cached = _load_cache()
    if cached is not None:
        X, y, ids, splits = cached
        print(f"Loaded cached features: {CACHE_FILE}  X={X.shape}")
        return X, y, ids, splits

    device = _device()
    print(f"Cache not found. Extracting Qwen features on {device}.")
    print(f"Data            : {DATA_FILE}")
    print(f"Aggregation mode: {AGGREGATION_MODE}")
    print(f"Spectral mode   : {SPECTRAL_MODE}")

    aggregation.AGGREGATION_MODE = AGGREGATION_MODE
    aggregation.SPECTRAL_MODE = SPECTRAL_MODE

    df = pd.read_csv(DATA_FILE)
    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    y = np.array([int(float(label)) for label in df["label"]])
    ids = df.index.to_numpy(dtype=np.int64)
    splits = split_data(y, df)

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    features: list[torch.Tensor] = []
    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Extracting & aggregating",
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
            feat = aggregation.aggregation_and_feature_extraction(
                hidden[i],
                mask[i],
                use_geometric=False,
            )
            features.append(feat.cpu())

    X = np.vstack([feature.numpy() for feature in features]).astype(np.float32)
    _save_cache(X, y, ids, splits)
    print(f"Saved cached features: {CACHE_FILE}  X={X.shape}")
    return X, y, ids, splits


def _hyperparameter_summary(probe_type: str) -> str:
    if probe_type in {"logreg", "logreg_l1"}:
        return f"LOGREG_C={os.getenv('LOGREG_C', '0.01')}"
    if probe_type == "mlp":
        return (
            f"MLP_HIDDEN={os.getenv('MLP_HIDDEN', '64')} "
            f"MLP_ALPHA={os.getenv('MLP_ALPHA', '0.001')}"
        )
    if probe_type in {"svm_linear", "svm_rbf"}:
        gamma = f" SVM_GAMMA={os.getenv('SVM_GAMMA', 'scale')}" if probe_type == "svm_rbf" else ""
        return f"SVM_C={os.getenv('SVM_C', '0.1' if probe_type == 'svm_linear' else '1.0')}{gamma}"
    if probe_type == "ridge":
        return f"RIDGE_ALPHA={os.getenv('RIDGE_ALPHA', '1.0')}"
    if probe_type == "knn":
        return (
            f"KNN_NEIGHBORS={os.getenv('KNN_NEIGHBORS', '5')} "
            f"KNN_WEIGHTS={os.getenv('KNN_WEIGHTS', 'distance')}"
        )
    return ""


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, _, splits = _extract_current_features()
    print(f"Feature matrix : {X.shape}")
    print(f"Splits         : {len(splits)}")

    rows: list[dict] = []
    for probe_type in PROBE_TYPES:
        for pca_components in PCA_COMPONENTS:
            os.environ["PROBE_TYPE"] = probe_type
            os.environ["PCA_N_COMPONENTS"] = pca_components
            os.environ["THRESHOLD_METRIC"] = THRESHOLD_METRIC

            t0 = time.time()
            row = {
                "aggregation_mode": AGGREGATION_MODE,
                "spectral_mode": SPECTRAL_MODE,
                "probe_type": probe_type,
                "pca_components": pca_components,
                "threshold_metric": THRESHOLD_METRIC,
                "val_accuracy": float("nan"),
                "val_f1": float("nan"),
                "val_auroc": float("nan"),
                "test_accuracy": float("nan"),
                "test_f1": float("nan"),
                "test_auroc": float("nan"),
                "best_threshold": float("nan"),
                "runtime_seconds": float("nan"),
                "status": "failed",
                "error_message": "",
            }

            print(
                f"\nRun: probe={probe_type}  PCA_N_COMPONENTS={pca_components}  "
                f"{_hyperparameter_summary(probe_type)}"
            )
            try:
                metrics = _run_probe_evaluation(X, y, splits)
                row.update(metrics)
                row["status"] = "ok"
                print(
                    f"  val_acc={metrics['val_accuracy']:.4f}  "
                    f"val_auroc={metrics['val_auroc']:.4f}  "
                    f"test_acc={metrics['test_accuracy']:.4f}  "
                    f"test_auroc={metrics['test_auroc']:.4f}  "
                    f"threshold={metrics['best_threshold']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001 - keep sweep robust.
                row["error_message"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                print(f"  FAILED: {row['error_message']}")
            finally:
                row["runtime_seconds"] = time.time() - t0
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    results = pd.DataFrame(rows)
    sorted_results = results.sort_values(
        by=["val_accuracy", "test_accuracy"],
        ascending=False,
        na_position="last",
    )
    results.to_csv(OUTPUT_FILE, index=False)
    sorted_results.to_csv(SORTED_OUTPUT_FILE, index=False)

    print("\nPCA/probe ablation summary")
    print(
        sorted_results[
            [
                "probe_type",
                "pca_components",
                "val_accuracy",
                "val_f1",
                "val_auroc",
                "test_accuracy",
                "test_f1",
                "test_auroc",
                "best_threshold",
                "status",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {SORTED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
