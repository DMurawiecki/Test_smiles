"""
Controlled response-token pooling stack/type ablation.

Everything except token pooling is fixed: original split protocol, layers
21-24 averaged first, train-only StandardScaler/PCA, and two classifiers
(linear SVM and the existing 2-hidden-layer BatchNorm+Dropout MLP).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aggregation  # noqa: E402
from experiments.run_mlp_vs_svm_trajectory import mlp_scores, train_mlp  # noqa: E402
from experiments.run_trajectory_features import DATA_FILE, _response_positions  # noqa: E402
from experiments.run_trajectory_range_threshold_ablation import find_best_threshold_by_f1  # noqa: E402
from model import MAX_LENGTH, _DEFAULT_MODEL, get_model_and_tokenizer  # noqa: E402
from splitting import split_data  # noqa: E402


BATCH_SIZE = 4
LAYERS_USED = (21, 22, 23, 24)
LAYER_AGGREGATION = "mean_21_22_23_24"
TOKEN_POOLING_STACKS = {
    "last_1": [1],
    "last_1_8_16_32": [1, 8, 16, 32],
    "last_1_3_5_8_16_32": [1, 3, 5, 8, 16, 32],
    "last_1_8_16_32_64_128_258": [1, 8, 16, 32, 64, 128, 258],
    "last_1_8_16_32_64_128_256": [1, 8, 16, 32, 64, 128, 256],
    "last_1_8_16_18_26_32": [1, 8, 16, 18, 26, 32],
}
POOLING_TYPES = (
    "mean",
    "max",
    "min",
    "mean_std",
    "mean_max",
    "mean_std_max",
    "mean_std_min_max",
    "ewma",
)
MLP2_HIDDEN = (128, 64)
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-3
MLP_MAX_EPOCHS = 300
MLP_PATIENCE = 30
MLP_BATCH_SIZE = 64


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_slug(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "p").replace("-", "_")


def _mean_layers_21_24(hidden_states: torch.Tensor) -> torch.Tensor:
    hs = aggregation._as_layer_tensor(hidden_states).detach()
    indices = [aggregation.transformer_layer_to_index(layer, hs) for layer in LAYERS_USED]
    return hs.index_select(0, torch.tensor(indices, device=hs.device)).float().mean(dim=0)


def _pool_window(X: np.ndarray, pooling_type: str, ewma_alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if pooling_type == "mean":
        return X.mean(axis=0)
    if pooling_type == "max":
        return X.max(axis=0)
    if pooling_type == "min":
        return X.min(axis=0)
    if pooling_type == "mean_std":
        return np.concatenate([X.mean(axis=0), X.std(axis=0, ddof=0)])
    if pooling_type == "mean_max":
        return np.concatenate([X.mean(axis=0), X.max(axis=0)])
    if pooling_type == "mean_std_max":
        return np.concatenate([X.mean(axis=0), X.std(axis=0, ddof=0), X.max(axis=0)])
    if pooling_type == "mean_std_min_max":
        return np.concatenate([X.mean(axis=0), X.std(axis=0, ddof=0), X.min(axis=0), X.max(axis=0)])
    if pooling_type == "ewma":
        positions = np.arange(X.shape[0], dtype=np.float64)
        weights = np.exp(float(ewma_alpha) * positions)
        weights = weights / weights.sum()
        return (weights[:, None] * X).sum(axis=0)
    raise ValueError(f"Unsupported pooling_type={pooling_type!r}")


def _feature_from_response_tokens(
    token_matrix: np.ndarray,
    windows: list[int],
    pooling_type: str,
    ewma_alpha: float,
) -> np.ndarray:
    parts = []
    n = token_matrix.shape[0]
    for k in windows:
        window = token_matrix[-min(int(k), n) :]
        parts.append(_pool_window(window, pooling_type, ewma_alpha))
    return np.concatenate(parts).astype(np.float32)


def _cache_path(output_dir: Path, model_name: str, stacks: list[str], pooling_types: list[str], ewma_alpha: float) -> Path:
    stack_slug = "-".join(stacks)
    pool_slug = "-".join(pooling_types)
    return output_dir / (
        f"token_pooling_cache_{_safe_slug(model_name)}_{stack_slug}_{pool_slug}_"
        f"ewma{str(ewma_alpha).replace('.', 'p')}.npz"
    )


def load_or_extract_features(
    output_dir: Path,
    model_name: str,
    stack_names: list[str],
    pooling_types: list[str],
    ewma_alpha: float,
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray, list]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(output_dir, model_name, stack_names, pooling_types, ewma_alpha)
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        X_by_combo = {
            (stack, pooling): np.asarray(data[f"X__{stack}__{pooling}"], dtype=np.float32)
            for stack in stack_names
            for pooling in pooling_types
        }
        y = np.asarray(data["y"], dtype=int)
        splits = []
        for i in range(int(data["n_splits"][0])):
            splits.append((
                np.asarray(data[f"split_{i}_train"], dtype=int),
                np.asarray(data[f"split_{i}_val"], dtype=int),
                np.asarray(data[f"split_{i}_test"], dtype=int),
            ))
        print(f"Loaded token pooling cache: {cache}")
        return X_by_combo, y, splits

    df = pd.read_csv(DATA_FILE)
    prompts = df["prompt"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()
    texts = [f"{p}{r}" for p, r in zip(prompts, responses)]
    y = df["label"].astype(float).astype(int).to_numpy()
    splits = split_data(y, df)

    device = _device()
    print(f"Cache not found. Extracting Qwen response-token features on {device}.")
    print(f"Model: {model_name}")
    print(f"Stacks: {', '.join(stack_names)}")
    print(f"Pooling types: {', '.join(pooling_types)}")
    model, tokenizer = get_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    buffers: dict[tuple[str, str], list[np.ndarray]] = {
        (stack, pooling): [] for stack in stack_names for pooling in pooling_types
    }
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting token pooling features", unit="batch"):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        masks = attention_mask.cpu()
        for i in range(hidden.size(0)):
            response_positions = _response_positions(tokenizer, prompts[start + i], texts[start + i], masks[i])
            token_repr = _mean_layers_21_24(hidden[i]).index_select(0, response_positions.to(hidden.device))
            token_matrix = token_repr.detach().cpu().numpy().astype(np.float32)
            for stack in stack_names:
                windows = TOKEN_POOLING_STACKS[stack]
                for pooling in pooling_types:
                    buffers[(stack, pooling)].append(
                        _feature_from_response_tokens(token_matrix, windows, pooling, ewma_alpha)
                    )

    X_by_combo = {
        combo: np.vstack(values).astype(np.float32)
        for combo, values in buffers.items()
    }
    payload: dict[str, object] = {
        "y": y,
        "n_splits": np.asarray([len(splits)], dtype=np.int64),
        "metadata": np.asarray([
            json.dumps({
                "model_name": model_name,
                "layers_used": list(LAYERS_USED),
                "layer_aggregation": LAYER_AGGREGATION,
                "stacks": stack_names,
                "pooling_types": pooling_types,
                "ewma_alpha": ewma_alpha,
            })
        ]),
    }
    for (stack, pooling), X in X_by_combo.items():
        payload[f"X__{stack}__{pooling}"] = X
    for i, (tr, va, te) in enumerate(splits):
        payload[f"split_{i}_train"] = tr
        payload[f"split_{i}_val"] = np.asarray([], dtype=np.int64) if va is None else va
        payload[f"split_{i}_test"] = te
    np.savez_compressed(cache, **payload)
    print(f"Saved token pooling cache: {cache}")
    return X_by_combo, y, splits


def _fit_transform_pca(
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])
    X_val_scaled = scaler.transform(X[val_idx])
    X_test_scaled = scaler.transform(X[test_idx])
    n = min(int(pca_components), X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
    pca = PCA(n_components=n, random_state=seed)
    return (
        pca.fit_transform(X_train_scaled).astype(np.float32),
        pca.transform(X_val_scaled).astype(np.float32),
        pca.transform(X_test_scaled).astype(np.float32),
        int(n),
    )


def _metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = float("nan")
    try:
        ap = average_precision_score(y_true, scores)
    except ValueError:
        ap = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _mean_numeric(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def evaluate_combo(
    X: np.ndarray,
    y: np.ndarray,
    splits: list,
    model_name: str,
    stack_name: str,
    pooling_type: str,
    pca_components: int,
    svm_c: float,
    seed: int,
) -> dict:
    rows = []
    for fold, (tr, va, te) in enumerate(splits, 1):
        if va is None:
            raise ValueError("Validation split is required for threshold tuning.")
        X_train, X_val, X_test, pca_dim = _fit_transform_pca(X, tr, va, te, pca_components, seed)
        if model_name == "svm_linear":
            model = SVC(C=svm_c, kernel="linear", probability=False, random_state=seed)
            model.fit(X_train, y[tr])
            val_scores = model.decision_function(X_val)
            test_scores = model.decision_function(X_test)
        elif model_name == "mlp2_bn_dropout":
            model = train_mlp(
                X_train,
                y[tr],
                X_val,
                y[va],
                MLP2_HIDDEN,
                MLP_DROPOUT,
                MLP_LR,
                MLP_WEIGHT_DECAY,
                MLP_MAX_EPOCHS,
                MLP_PATIENCE,
                MLP_BATCH_SIZE,
                seed + fold,
            )
            val_scores = mlp_scores(model, X_val)
            test_scores = mlp_scores(model, X_test)
        else:
            raise ValueError(f"Unknown model_name={model_name!r}")
        best_threshold, _ = find_best_threshold_by_f1(y[va], val_scores)
        row = _metric_row(y[te], test_scores, best_threshold)
        row.update({
            "fold": fold,
            "best_threshold": float(best_threshold),
            "train_size": len(tr),
            "val_size_or_test_size": len(te),
            "raw_feature_dim": X.shape[1],
            "pca_components": pca_dim,
        })
        rows.append(row)

    out = {
        "model_name": model_name,
        "token_pooling_stack_name": stack_name,
        "token_pooling_windows": ",".join(str(v) for v in TOKEN_POOLING_STACKS[stack_name]),
        "pooling_type": pooling_type,
        "layers_used": ",".join(str(v) for v in LAYERS_USED),
        "layer_aggregation": LAYER_AGGREGATION,
        "svm_c": float(svm_c) if model_name == "svm_linear" else float("nan"),
        "seed": seed,
    }
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "best_threshold",
        "train_size",
        "val_size_or_test_size",
        "raw_feature_dim",
        "pca_components",
    ):
        out[key] = _mean_numeric(rows, key)
    for key in ("tn", "fp", "fn", "tp"):
        out[key] = int(sum(int(row[key]) for row in rows))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=_DEFAULT_MODEL)
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--stacks", default=None, help="Comma-separated stack names.")
    parser.add_argument("--pooling-types", default=None, help="Comma-separated pooling types.")
    parser.add_argument("--ewma-alpha", type=float, default=0.1)
    args = parser.parse_args()

    stack_names = _parse_csv_list(args.stacks, list(TOKEN_POOLING_STACKS))
    pooling_types = _parse_csv_list(args.pooling_types, list(POOLING_TYPES))
    bad_stacks = [name for name in stack_names if name not in TOKEN_POOLING_STACKS]
    bad_pooling = [name for name in pooling_types if name not in POOLING_TYPES]
    if bad_stacks:
        raise ValueError(f"Unknown stack names {bad_stacks}; expected {list(TOKEN_POOLING_STACKS)}")
    if bad_pooling:
        raise ValueError(f"Unknown pooling types {bad_pooling}; expected {list(POOLING_TYPES)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X_by_combo, y, splits = load_or_extract_features(
        args.output_dir,
        args.model_name,
        stack_names,
        pooling_types,
        args.ewma_alpha,
    )
    print(f"Samples: {len(y)}")
    print(f"Split folds: {len(splits)}")
    rows = []
    t0 = time.time()
    for stack in stack_names:
        for pooling in pooling_types:
            X = X_by_combo[(stack, pooling)]
            if not np.isfinite(X).all():
                print(f"WARNING: non-finite values in {stack}/{pooling}; replacing with zeros.")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"\nFeature matrix {stack} + {pooling}: {X.shape}")
            for model_name in ("svm_linear", "mlp2_bn_dropout"):
                row = evaluate_combo(
                    X,
                    y,
                    splits,
                    model_name,
                    stack,
                    pooling,
                    args.pca_components,
                    args.svm_c,
                    args.seed,
                )
                rows.append(row)
                print(
                    f"  {model_name}: roc_auc={row['roc_auc']:.4f} "
                    f"ap={row['average_precision']:.4f} f1={row['f1']:.4f} "
                    f"bal_acc={row['balanced_accuracy']:.4f}"
                )

    result = pd.DataFrame(rows)
    full_path = args.output_dir / "token_pooling_stack_and_type_ablation.csv"
    sorted_path = args.output_dir / "token_pooling_stack_and_type_ablation_sorted.csv"
    result.to_csv(full_path, index=False)
    sorted_result = result.sort_values(
        ["roc_auc", "f1", "balanced_accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    sorted_result.insert(0, "rank", np.arange(1, len(sorted_result) + 1))
    sorted_result.to_csv(sorted_path, index=False)

    leaderboard_cols = [
        "rank",
        "model_name",
        "token_pooling_stack_name",
        "pooling_type",
        "raw_feature_dim",
        "pca_components",
        "roc_auc",
        "average_precision",
        "f1",
        "balanced_accuracy",
        "precision",
        "recall",
    ]
    print("\nToken pooling stack/type leaderboard")
    print(sorted_result[leaderboard_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved full results to {full_path}")
    print(f"Saved sorted results to {sorted_path}")
    print(f"Runtime seconds: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
