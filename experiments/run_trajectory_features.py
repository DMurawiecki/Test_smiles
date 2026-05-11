"""
Layer Trajectory Geometry Probe experiment.

This script keeps the repository split protocol intact and evaluates whether
cross-layer dynamics features add signal beyond the shared response-aware
baseline: concat(mean(21-24) over response last-token/last8/last16/last32).

Layer convention: cached trajectory tensors contain transformer layers 1..24
only, with array index 0 corresponding to transformer layer 1.  The embedding
hidden state, when present in model outputs, is skipped.
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
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

import aggregation
from model import MAX_LENGTH, get_model_and_tokenizer
from splitting import split_data


DATA_FILE = ROOT / "data" / "dataset.csv"
BATCH_SIZE = 4
BASELINE_AGGREGATION_MODE = "mean_21_24"
BASELINE_SPECTRAL_MODE = "none"
N_TRANSFORMER_LAYERS = 24
DEFAULT_POOLING_MODES = (
    "response_last_token",
    "response_last_8_mean",
    "response_last_16_mean",
    "response_last_32_mean",
)
POOLING_MODE_SHORT = {
    "response_last_token": "last_token",
    "response_last_8_mean": "last8",
    "response_last_16_mean": "last16",
    "response_last_32_mean": "last32",
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_range(value: str) -> tuple[int, int]:
    start, end = value.split("-", 1)
    start_i, end_i = int(start), int(end)
    if start_i < 1 or end_i < start_i or end_i > N_TRANSFORMER_LAYERS:
        raise ValueError(f"Invalid layer range {value!r}; expected e.g. 1-24.")
    return start_i, end_i


def _real_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.detach().reshape(-1)
    positions = torch.nonzero(mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if positions.numel() == 0:
        return torch.tensor([mask.numel() - 1], device=mask.device, dtype=torch.long)
    return positions.to(dtype=torch.long)


def _tail_mean(sequence_repr: torch.Tensor, positions: torch.Tensor, width: int) -> torch.Tensor:
    real_sequence = sequence_repr.index_select(0, positions.to(sequence_repr.device))
    tail = real_sequence[-min(width, real_sequence.size(0)) :]
    return tail.to(dtype=torch.float32).mean(dim=0)


def _response_positions(
    tokenizer,
    prompt: str,
    full_text: str,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    real_positions = _real_token_positions(attention_mask)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    start = min(len(prompt_ids), max(len(full_ids) - 1, 0), int(real_positions[-1].item()))
    end = min(len(full_ids), int(real_positions[-1].item()) + 1)
    if end <= start:
        return real_positions[-1:]
    return torch.arange(start, end, dtype=torch.long)


def _pooling_width(pooling_mode: str, token_window: int) -> int:
    if pooling_mode == "response_last_token":
        return 1
    if pooling_mode == "response_last_8_mean":
        return 8
    if pooling_mode == "response_last_16_mean":
        return 16
    if pooling_mode == "response_last_32_mean":
        return 32
    if pooling_mode == "response_last_k_mean":
        return token_window
    raise ValueError(f"Unsupported pooling mode {pooling_mode!r}.")


def _select_response_positions(response_positions: torch.Tensor, pooling_mode: str, token_window: int) -> torch.Tensor:
    width = _pooling_width(pooling_mode, token_window)
    return response_positions[-min(width, response_positions.numel()) :]


def _layer_vectors_for_positions(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Return pooled vectors for transformer layers 1..24, shape [24, hidden_dim]."""
    hs = aggregation._as_layer_tensor(hidden_states).detach()
    positions = positions.to(device=hs.device)
    vectors: list[torch.Tensor] = []
    for layer_number in range(1, N_TRANSFORMER_LAYERS + 1):
        idx = aggregation.transformer_layer_to_index(layer_number, hs)
        vectors.append(hs[idx].index_select(0, positions).float().mean(dim=0))
    return torch.stack(vectors, dim=0).to(dtype=torch.float32)


def _baseline_from_layer_vectors_by_mode(
    layer_vectors_by_mode: dict[str, np.ndarray],
    pooling_modes: list[str],
) -> np.ndarray:
    parts = [layer_vectors_by_mode[mode][:, 20:24, :].mean(axis=1) for mode in pooling_modes]
    return np.concatenate(parts, axis=1).astype(np.float32)


def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return 0.0
    return float(np.dot(a, b) / (denom + eps))


def _segment_means(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    chunks = np.array_split(values, 3)
    return tuple(float(chunk.mean()) if chunk.size else 0.0 for chunk in chunks)


def compute_trajectory_features(
    layer_vectors: np.ndarray,
    layer_start: int,
    layer_end: int,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute scalar trajectory geometry features for a layer range.

    ``layer_vectors`` is indexed as transformer layers 1..24, so layer 1 lives
    at row 0.  Returned feature values are finite floats; curvature features are
    zero-filled when the range is too short for adjacent-delta comparisons.
    """
    Z = np.asarray(layer_vectors[layer_start - 1 : layer_end], dtype=np.float64)
    prefix = f"traj_{layer_start}_{layer_end}__"
    features: dict[str, float] = {}

    if Z.shape[0] < 2:
        names = _trajectory_feature_names(layer_start, layer_end)
        return {name: 0.0 for name in names}

    deltas = Z[1:] - Z[:-1]
    step_norms = np.linalg.norm(deltas, axis=1)
    early_step, mid_step, late_step = _segment_means(step_norms)

    features[prefix + "step_mean"] = float(step_norms.mean())
    features[prefix + "step_std"] = float(step_norms.std())
    features[prefix + "step_max"] = float(step_norms.max())
    features[prefix + "step_min"] = float(step_norms.min())
    features[prefix + "step_median"] = float(np.median(step_norms))
    features[prefix + "step_last"] = float(step_norms[-1])
    features[prefix + "step_early_mean"] = early_step
    features[prefix + "step_mid_mean"] = mid_step
    features[prefix + "step_late_mean"] = late_step
    features[prefix + "early_late_step_ratio"] = float(early_step / (late_step + eps))

    path_length = float(step_norms.sum())
    endpoint_distance = float(np.linalg.norm(Z[-1] - Z[0]))
    features[prefix + "path_length"] = path_length
    features[prefix + "endpoint_distance"] = endpoint_distance
    features[prefix + "straightness"] = float(endpoint_distance / (path_length + eps))

    cos_steps = np.asarray([_safe_cosine(Z[i], Z[i - 1], eps) for i in range(1, Z.shape[0])])
    early_cos, mid_cos, late_cos = _segment_means(cos_steps)
    features[prefix + "cos_step_mean"] = float(cos_steps.mean())
    features[prefix + "cos_step_std"] = float(cos_steps.std())
    features[prefix + "cos_step_min"] = float(cos_steps.min())
    features[prefix + "cos_step_max"] = float(cos_steps.max())
    features[prefix + "cos_first_last"] = _safe_cosine(Z[0], Z[-1], eps)
    features[prefix + "cos_early_mean"] = early_cos
    features[prefix + "cos_mid_mean"] = mid_cos
    features[prefix + "cos_late_mean"] = late_cos

    k = min(4, Z.shape[0])
    late_before_final = Z[-k:-1]
    if late_before_final.size:
        late_l2 = np.linalg.norm(late_before_final - Z[-1], axis=1)
        late_cos = np.asarray([_safe_cosine(z, Z[-1], eps) for z in late_before_final])
        features[prefix + "late_l2_to_final_mean"] = float(late_l2.mean())
        features[prefix + "late_l2_to_final_max"] = float(late_l2.max())
        features[prefix + "late_cos_to_final_mean"] = float(late_cos.mean())
        features[prefix + "late_cos_to_final_min"] = float(late_cos.min())
    else:
        features[prefix + "late_l2_to_final_mean"] = 0.0
        features[prefix + "late_l2_to_final_max"] = 0.0
        features[prefix + "late_cos_to_final_mean"] = 0.0
        features[prefix + "late_cos_to_final_min"] = 0.0

    if deltas.shape[0] >= 2:
        delta_cos = np.asarray(
            [_safe_cosine(deltas[i], deltas[i + 1], eps) for i in range(deltas.shape[0] - 1)]
        )
        curvature = 1.0 - delta_cos
        early_curv, mid_curv, late_curv = _segment_means(curvature)
        features[prefix + "delta_cos_mean"] = float(delta_cos.mean())
        features[prefix + "delta_cos_std"] = float(delta_cos.std())
        features[prefix + "delta_cos_min"] = float(delta_cos.min())
        features[prefix + "delta_cos_max"] = float(delta_cos.max())
        features[prefix + "curvature_mean"] = float(curvature.mean())
        features[prefix + "curvature_max"] = float(curvature.max())
        features[prefix + "early_curvature_mean"] = early_curv
        features[prefix + "mid_curvature_mean"] = mid_curv
        features[prefix + "late_curvature_mean"] = late_curv
    else:
        for name in (
            "delta_cos_mean",
            "delta_cos_std",
            "delta_cos_min",
            "delta_cos_max",
            "curvature_mean",
            "curvature_max",
            "early_curvature_mean",
            "mid_curvature_mean",
            "late_curvature_mean",
        ):
            features[prefix + name] = 0.0

    return {key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key, value in features.items()}


def _trajectory_feature_names(layer_start: int, layer_end: int) -> list[str]:
    suffixes = (
        "step_mean",
        "step_std",
        "step_max",
        "step_min",
        "step_median",
        "step_last",
        "step_early_mean",
        "step_mid_mean",
        "step_late_mean",
        "early_late_step_ratio",
        "path_length",
        "endpoint_distance",
        "straightness",
        "cos_step_mean",
        "cos_step_std",
        "cos_step_min",
        "cos_step_max",
        "cos_first_last",
        "cos_early_mean",
        "cos_mid_mean",
        "cos_late_mean",
        "late_l2_to_final_mean",
        "late_l2_to_final_max",
        "late_cos_to_final_mean",
        "late_cos_to_final_min",
        "delta_cos_mean",
        "delta_cos_std",
        "delta_cos_min",
        "delta_cos_max",
        "curvature_mean",
        "curvature_max",
        "early_curvature_mean",
        "mid_curvature_mean",
        "late_curvature_mean",
    )
    return [f"traj_{layer_start}_{layer_end}__{suffix}" for suffix in suffixes]


def build_trajectory_feature_matrix(
    all_layer_vectors: np.ndarray,
    layer_range: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    """Build trajectory feature matrix for all samples."""
    start, end = layer_range
    names = _trajectory_feature_names(start, end)
    rows = []
    for sample_layers in all_layer_vectors:
        feature_dict = compute_trajectory_features(sample_layers, start, end)
        rows.append([feature_dict[name] for name in names])
    return np.asarray(rows, dtype=np.float32), names


def _replace_prefix(names: list[str], old_prefix: str, new_prefix: str) -> list[str]:
    return [new_prefix + name[len(old_prefix) :] if name.startswith(old_prefix) else name for name in names]


def build_pooled_trajectory_feature_matrix(
    layer_vectors_by_mode: dict[str, np.ndarray],
    pooling_modes: list[str],
    layer_range: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    old_prefix = f"traj_{start}_{end}__"
    matrices, names = [], []
    for mode in pooling_modes:
        X, mode_names = build_trajectory_feature_matrix(layer_vectors_by_mode[mode], layer_range)
        matrices.append(X)
        new_prefix = f"trajectory_{POOLING_MODE_SHORT[mode]}_{start}_{end}__"
        names.extend(_replace_prefix(mode_names, old_prefix, new_prefix))
    return np.concatenate(matrices, axis=1).astype(np.float32), names


def _cache_file(output_dir: Path, token_window: int, pooling_modes: list[str]) -> Path:
    slug = "_".join(POOLING_MODE_SHORT.get(mode, mode).replace("response_", "") for mode in pooling_modes)
    return output_dir / f"trajectory_cache_tail{token_window}_{slug}_response_mean21_24.npz"


def load_or_extract_representations(
    output_dir: Path,
    token_window: int,
    pooling_modes: list[str] | None = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
]:
    """Load or create baseline hidden features and per-layer pooled vectors."""
    pooling_modes = list(DEFAULT_POOLING_MODES if pooling_modes is None else pooling_modes)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file(output_dir, token_window, pooling_modes)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        X_hidden = np.asarray(data["X_hidden"], dtype=np.float32)
        layer_vectors = {
            mode: np.asarray(data[f"layer_vectors__{mode}"], dtype=np.float32)
            for mode in pooling_modes
        }
        y = np.asarray(data["y"], dtype=int)
        splits = []
        for i in range(int(data["n_splits"][0])):
            tr = np.asarray(data[f"split_{i}_train"], dtype=int)
            va_raw = np.asarray(data[f"split_{i}_val"], dtype=int)
            va = None if va_raw.size == 0 else va_raw
            te = np.asarray(data[f"split_{i}_test"], dtype=int)
            splits.append((tr, va, te))
        print(f"Loaded trajectory cache: {cache_path}")
        return X_hidden, layer_vectors, y, splits

    device = _device()
    print(f"Cache not found. Extracting Qwen hidden states on {device}.")
    print(f"Response pooling modes: {', '.join(pooling_modes)}")

    df = pd.read_csv(DATA_FILE)
    prompts = df["prompt"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()
    texts = [f"{prompt}{response}" for prompt, response in zip(prompts, responses)]
    y = np.asarray([int(float(label)) for label in df["label"]], dtype=int)
    splits = split_data(y, df)

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    layer_features: dict[str, list[torch.Tensor]] = {mode: [] for mode in pooling_modes}
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting representations", unit="batch"):
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
            all_response_positions = _response_positions(
                tokenizer,
                prompts[start + i],
                texts[start + i],
                mask[i],
            )
            for mode in pooling_modes:
                positions = _select_response_positions(all_response_positions, mode, token_window)
                layer_features[mode].append(_layer_vectors_for_positions(hidden[i], positions).cpu())

    layer_vectors = {
        mode: np.stack([feature.numpy() for feature in features]).astype(np.float32)
        for mode, features in layer_features.items()
    }
    X_hidden = _baseline_from_layer_vectors_by_mode(layer_vectors, pooling_modes)

    payload: dict[str, object] = {
        "X_hidden": X_hidden,
        "y": y,
        "n_splits": np.asarray([len(splits)], dtype=np.int64),
        "metadata": np.asarray(
            [
                json.dumps(
                    {
                        "pooling_modes": pooling_modes,
                        "baseline_definition": (
                            "concat(mean21_24 over response_last_token, response_last_8_mean, "
                            "response_last_16_mean, response_last_32_mean)"
                        ),
                        "handcrafted_blocks_use_pca": False,
                    }
                )
            ]
        ),
    }
    for mode, values in layer_vectors.items():
        payload[f"layer_vectors__{mode}"] = values
    for i, (tr, va, te) in enumerate(splits):
        payload[f"split_{i}_train"] = tr
        payload[f"split_{i}_val"] = np.asarray([], dtype=np.int64) if va is None else va
        payload[f"split_{i}_test"] = te
    np.savez_compressed(cache_path, **payload)
    print(f"Saved trajectory cache: {cache_path}")
    return X_hidden, layer_vectors, y, splits


class SplitFeatureBuilder:
    """Fit train-only preprocessing and transform train/val/test matrices."""

    def __init__(self, pca_components: int, use_hidden: bool, use_trajectory: bool) -> None:
        self.pca_components = pca_components
        self.use_hidden = use_hidden
        self.use_trajectory = use_trajectory
        self.hidden_scaler = StandardScaler()
        self.pca: PCA | None = None
        self.trajectory_scaler = StandardScaler()

    def fit_transform_train(self, X_hidden: np.ndarray, X_traj: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.use_hidden:
            hidden_scaled = self.hidden_scaler.fit_transform(X_hidden)
            n_components = min(self.pca_components, hidden_scaled.shape[0] - 1, hidden_scaled.shape[1])
            self.pca = PCA(n_components=n_components, random_state=42)
            parts.append(self.pca.fit_transform(hidden_scaled))
        if self.use_trajectory:
            parts.append(self.trajectory_scaler.fit_transform(X_traj))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform(self, X_hidden: np.ndarray, X_traj: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.use_hidden:
            if self.pca is None:
                raise RuntimeError("Hidden PCA has not been fitted.")
            hidden_scaled = self.hidden_scaler.transform(X_hidden)
            parts.append(self.pca.transform(hidden_scaled))
        if self.use_trajectory:
            parts.append(self.trajectory_scaler.transform(X_traj))
        return np.concatenate(parts, axis=1).astype(np.float32)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
    }


def evaluate_experiment(
    name: str,
    X_hidden: np.ndarray,
    X_traj: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
    use_hidden: bool,
    use_trajectory: bool,
    pca_components: int,
    svm_c: float,
    seed: int,
) -> tuple[dict, pd.DataFrame, np.ndarray | None, list[str]]:
    fold_rows: list[dict] = []
    test_predictions: list[pd.DataFrame] = []
    feature_names: list[str] = []
    last_model: SVC | None = None
    last_X_val: np.ndarray | None = None
    last_y_val: np.ndarray | None = None

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits):
        builder = SplitFeatureBuilder(pca_components, use_hidden, use_trajectory)
        X_train = builder.fit_transform_train(X_hidden[idx_train], X_traj[idx_train])
        X_test = builder.transform(X_hidden[idx_test], X_traj[idx_test])
        X_val = builder.transform(X_hidden[idx_val], X_traj[idx_val]) if idx_val is not None else None

        if not feature_names:
            names: list[str] = []
            if use_hidden:
                n_pca = builder.pca.n_components_ if builder.pca is not None else 0
                names.extend([f"pca_{i}" for i in range(n_pca)])
            if use_trajectory:
                names.extend([f"trajectory_{i}" for i in range(X_traj.shape[1])])
            feature_names = names

        model = SVC(C=svm_c, kernel="linear", probability=False, random_state=seed)
        model.fit(X_train, y[idx_train])

        row: dict = {
            "fold": fold_idx + 1,
            "train_size": len(idx_train),
            "val_size": 0 if idx_val is None else len(idx_val),
            "test_size": len(idx_test),
            "final_feature_dim": X_train.shape[1],
        }
        if X_val is not None:
            val_pred = model.predict(X_val)
            val_scores = model.decision_function(X_val)
            row.update({f"val_{k}": v for k, v in _metrics(y[idx_val], val_pred, val_scores).items()})
            last_X_val = X_val
            last_y_val = y[idx_val]

        test_pred = model.predict(X_test)
        test_scores = model.decision_function(X_test)
        row.update({f"test_{k}": v for k, v in _metrics(y[idx_test], test_pred, test_scores).items()})
        fold_rows.append(row)
        test_predictions.append(
            pd.DataFrame(
                {
                    "fold": fold_idx + 1,
                    "id": idx_test,
                    "label": y[idx_test],
                    "prediction": test_pred,
                    "score": test_scores,
                }
            )
        )
        last_model = model

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "experiment": name,
        "use_hidden": use_hidden,
        "use_trajectory": use_trajectory,
        "pca_components": pca_components if use_hidden else 0,
        "svm_c": svm_c,
        "final_feature_dim": float(fold_df["final_feature_dim"].mean()),
    }
    for split in ("val", "test"):
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            col = f"{split}_{metric}"
            summary[col] = float(fold_df[col].mean()) if col in fold_df else float("nan")

    predictions = pd.concat(test_predictions, ignore_index=True) if test_predictions else None
    importance_matrix = None
    if last_model is not None and last_X_val is not None and last_y_val is not None:
        importance_matrix = (last_model, last_X_val, last_y_val)
    return summary, fold_df, predictions, feature_names, importance_matrix


def _class_distribution(y: np.ndarray, idx: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y[idx], return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "trajectory_features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectory-ranges", nargs="+", default=["21-24", "10-24", "1-24"])
    parser.add_argument("--token-window", type=int, default=32)
    parser.add_argument("--pooling-modes", nargs="+", default=list(DEFAULT_POOLING_MODES))
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--permutation-importance", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "baseline_definition": (
                "concat(mean21_24 over response_last_token, response_last_8_mean, "
                "response_last_16_mean, response_last_32_mean)"
            ),
            "hidden_pca_components": args.pca_components,
            "handcrafted_blocks_use_pca": False,
        }
    )
    with (output_dir / "config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2)

    t0 = time.time()
    unsupported = [mode for mode in args.pooling_modes if mode not in POOLING_MODE_SHORT]
    if unsupported:
        raise ValueError(f"Unsupported --pooling-modes {unsupported}; expected {list(POOLING_MODE_SHORT)}")
    X_hidden, all_layer_vectors, y, splits = load_or_extract_representations(
        output_dir,
        args.token_window,
        args.pooling_modes,
    )
    print(f"Samples: {len(y)}")
    for mode in args.pooling_modes:
        pooled_hidden = all_layer_vectors[mode][:, 20:24, :].mean(axis=1)
        print(f"Baseline pooled hidden block {mode}: {pooled_hidden.shape}")
    print(f"Final X_hidden shape before PCA: {X_hidden.shape}")
    for mode in args.pooling_modes:
        print(f"Layer vector tensor {mode}: {all_layer_vectors[mode].shape}")
    for i, (tr, va, te) in enumerate(splits, 1):
        val_dist = {} if va is None else _class_distribution(y, va)
        print(
            f"Fold {i}: train={len(tr)} { _class_distribution(y, tr) }  "
            f"val={0 if va is None else len(va)} {val_dist}  "
            f"test={len(te)} { _class_distribution(y, te) }"
        )

    parsed_ranges = {value: _parse_range(value) for value in args.trajectory_ranges}
    trajectory_matrices: dict[str, np.ndarray] = {}
    trajectory_names: dict[str, list[str]] = {}
    for label, layer_range in parsed_ranges.items():
        X_traj, names = build_pooled_trajectory_feature_matrix(all_layer_vectors, args.pooling_modes, layer_range)
        trajectory_matrices[label] = X_traj
        trajectory_names[label] = names
        print(f"Trajectory range {label}: feature shape {X_traj.shape}")

    zero_traj = np.zeros((len(y), 0), dtype=np.float32)
    experiments: list[tuple[str, bool, bool, str | None]] = [
        ("E0_baseline_only", True, False, None),
        ("E1_trajectory_only_1_24", False, True, "1-24"),
        ("E2_baseline_plus_trajectory_21_24", True, True, "21-24"),
        ("E3_baseline_plus_trajectory_10_24", True, True, "10-24"),
        ("E4_baseline_plus_trajectory_1_24", True, True, "1-24"),
    ]

    summaries: list[dict] = []
    all_feature_names: dict[str, list[str]] = {}
    fold_details: dict[str, list[dict]] = {}
    e4_importance = None

    for name, use_hidden, use_traj, range_label in experiments:
        X_traj = trajectory_matrices[range_label] if range_label is not None else zero_traj
        print(f"\n{name}")
        print(f"  hidden input shape: {X_hidden.shape if use_hidden else (len(y), 0)}")
        print(f"  trajectory input shape: {X_traj.shape if use_traj else (len(y), 0)}")
        summary, fold_df, predictions, feature_names, importance_payload = evaluate_experiment(
            name,
            X_hidden,
            X_traj,
            y,
            splits,
            use_hidden,
            use_traj,
            args.pca_components,
            args.svm_c,
            args.seed,
        )
        summary["trajectory_range"] = range_label or "none"
        summary["runtime_seconds_total"] = time.time() - t0
        summaries.append(summary)
        fold_details[name] = fold_df.to_dict(orient="records")

        if use_traj and range_label is not None:
            pca_count = args.pca_components if use_hidden else 0
            all_feature_names[name] = [f"pca_{i}" for i in range(pca_count)] + trajectory_names[range_label]
        else:
            all_feature_names[name] = feature_names

        print(
            f"  final feature dim ~= {summary['final_feature_dim']:.0f}  "
            f"val_f1={summary['val_f1']:.4f}  val_acc={summary['val_accuracy']:.4f}  "
            f"test_f1={summary['test_f1']:.4f}  test_acc={summary['test_accuracy']:.4f}"
        )

        if not args.no_predictions and predictions is not None:
            predictions.to_csv(output_dir / f"predictions_{name}.csv", index=False)

        if name.startswith("E4_"):
            e4_importance = (importance_payload, all_feature_names[name])

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "summary.json").open("w") as f:
        json.dump({"summary": summaries, "folds": fold_details}, f, indent=2)
    with (output_dir / "feature_names.json").open("w") as f:
        json.dump(all_feature_names, f, indent=2)

    if args.permutation_importance and e4_importance is not None:
        payload, feature_names = e4_importance
        if payload is not None:
            model, X_val, y_val = payload
            result = permutation_importance(
                model,
                X_val,
                y_val,
                scoring="f1",
                n_repeats=10,
                random_state=args.seed,
            )
            pd.DataFrame(
                {
                    "feature_name": feature_names,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            ).sort_values("importance_mean", ascending=False).to_csv(
                output_dir / "permutation_importance.csv",
                index=False,
            )

    print("\nLayer Trajectory Geometry Probe summary")
    print(
        summary_df[
            [
                "experiment",
                "trajectory_range",
                "final_feature_dim",
                "val_accuracy",
                "val_f1",
                "val_roc_auc",
                "test_accuracy",
                "test_f1",
                "test_roc_auc",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    best_val = summary_df.sort_values("val_f1", ascending=False).iloc[0]
    best_test = summary_df.sort_values("test_f1", ascending=False).iloc[0]
    e0 = summary_df.loc[summary_df["experiment"] == "E0_baseline_only"].iloc[0]
    e4 = summary_df.loc[summary_df["experiment"] == "E4_baseline_plus_trajectory_1_24"].iloc[0]
    print(f"\nBest by validation F1: {best_val['experiment']} ({best_val['val_f1']:.4f})")
    print(f"Best by internal test F1: {best_test['experiment']} ({best_test['test_f1']:.4f})")
    print(
        "Delta E4 - E0: "
        f"val_f1={e4['val_f1'] - e0['val_f1']:.4f}, "
        f"test_f1={e4['test_f1'] - e0['test_f1']:.4f}, "
        f"val_acc={e4['val_accuracy'] - e0['val_accuracy']:.4f}, "
        f"test_acc={e4['test_accuracy'] - e0['test_accuracy']:.4f}"
    )
    print(f"\nSaved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
