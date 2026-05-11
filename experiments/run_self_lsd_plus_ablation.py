"""
Reference-free Self-LSD++ ablation for hallucination detection.

No gold/reference answers are required.  For each sample, the model response's
own late-layer representation acts as an anchor.  The experiment keeps the
existing split protocol and preprocessing style: hidden baseline features get
train-only scaling + PCA, every non-hidden block gets its own train-only
StandardScaler, blocks are concatenated, and a linear SVM is evaluated with a
validation-tuned F1 threshold.

Layer convention: cached index 0 is transformer layer 1.  When model outputs
include embeddings, hidden_states[1] is transformer layer 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
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

import aggregation  # noqa: E402
from experiments.run_trajectory_features import (  # noqa: E402
    DATA_FILE,
    N_TRANSFORMER_LAYERS,
    _class_distribution,
    _parse_range,
    _safe_cosine,
    build_trajectory_feature_matrix,
)
from model import MAX_LENGTH, _DEFAULT_MODEL, get_model_and_tokenizer  # noqa: E402
from splitting import split_data  # noqa: E402


MODEL_NAME = _DEFAULT_MODEL
BATCH_SIZE = 4
BLOCKS = (
    "trajectory",
    "mozeel_lsd",
    "self_alignment",
    "directional_anchor",
    "progress_anchor",
    "orthogonal_drift",
    "prompt_response_coupling",
)


@dataclass
class FoldArtifact:
    fold: int
    model: SVC
    X_val: np.ndarray
    y_val: np.ndarray
    threshold: float
    feature_names: list[str]


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_cache_name() -> str:
    return MODEL_NAME.lower().replace("/", "_").replace(".", "p").replace("-", "_")


def _validate_dataset(
    prompt_column: str,
    answer_column: str,
    label_column: str,
    truthful_label: int,
    hallucinated_label: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_FILE)
    missing = [col for col in (prompt_column, answer_column, label_column) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Available: {df.columns.tolist()}")
    if df[[prompt_column, answer_column]].isna().any().any():
        raise ValueError("prompt/response columns contain null values.")

    raw = df[label_column].astype(float).astype(int).to_numpy()
    allowed = {truthful_label, hallucinated_label}
    observed = set(np.unique(raw).tolist())
    if not observed.issubset(allowed):
        raise ValueError(f"Unexpected labels {sorted(observed)}; expected subset of {sorted(allowed)}")
    y = (raw == hallucinated_label).astype(int)
    return df, raw, y


def _token_span(
    tokenizer,
    prompt: str,
    full_text: str,
    attention_mask: torch.Tensor,
    token_window: int,
    pooling: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    real_positions = torch.nonzero(attention_mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if real_positions.numel() == 0:
        real_positions = torch.tensor([attention_mask.numel() - 1], dtype=torch.long)
    if pooling == "full_last_k_mean":
        response_positions = real_positions
    else:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        start = min(len(prompt_ids), max(len(full_ids) - 1, 0), int(real_positions[-1].item()))
        end = min(len(full_ids), int(real_positions[-1].item()) + 1)
        if end <= start:
            response_positions = real_positions[-1:]
        else:
            response_positions = torch.arange(start, end, dtype=torch.long)
    response_positions = response_positions[-min(token_window, response_positions.numel()) :]
    prompt_positions = real_positions[real_positions < response_positions[0]]
    if prompt_positions.numel() == 0:
        prompt_positions = real_positions[:1]
    prompt_positions = prompt_positions[-min(token_window, prompt_positions.numel()) :]
    if pooling == "response_last_token":
        response_positions = response_positions[-1:]
    return response_positions, prompt_positions


def _pool_layers(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    hs = aggregation._as_layer_tensor(hidden_states).detach()
    vectors = []
    for layer_number in range(1, N_TRANSFORMER_LAYERS + 1):
        idx = aggregation.transformer_layer_to_index(layer_number, hs)
        vectors.append(hs[idx].index_select(0, positions.to(hs.device)).float().mean(dim=0))
    return torch.stack(vectors, dim=0)


def _baseline_from_response_layers(response_layers: np.ndarray) -> np.ndarray:
    # Match aggregation.py's mean_21_24 hidden pooling shape only for the layer
    # average vector; diagnostics are omitted because this script's baseline is
    # explicitly "mean(21-24) response hidden -> PCA".
    return response_layers[:, 20:24, :].mean(axis=1).astype(np.float32)


def _cache_file(output_dir: Path, token_window: int, pooling: str, prompt_col: str, answer_col: str) -> Path:
    return output_dir / (
        f"self_lsd_cache_tail{token_window}_{pooling}_{_model_cache_name()}_"
        f"{prompt_col}_{answer_col}.npz"
    )


def load_or_extract_self_lsd_cache(
    output_dir: Path,
    token_window: int,
    pooling: str,
    prompt_column: str,
    answer_column: str,
    label_column: str,
    truthful_label: int,
    hallucinated_label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file(output_dir, token_window, pooling, prompt_column, answer_column)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        splits = []
        for i in range(int(data["n_splits"][0])):
            tr = np.asarray(data[f"split_{i}_train"], dtype=int)
            va = np.asarray(data[f"split_{i}_val"], dtype=int)
            te = np.asarray(data[f"split_{i}_test"], dtype=int)
            splits.append((tr, va, te))
        print(f"Loaded Self-LSD++ cache: {cache_path}")
        return (
            np.asarray(data["baseline_hidden"], dtype=np.float32),
            np.asarray(data["response_layer_vectors"], dtype=np.float32),
            np.asarray(data["prompt_layer_vectors"], dtype=np.float32),
            np.asarray(data["raw_label"], dtype=int),
            np.asarray(data["y"], dtype=int),
            splits,
        )

    df, raw_label, y = _validate_dataset(
        prompt_column, answer_column, label_column, truthful_label, hallucinated_label
    )
    splits = split_data(y, df)
    prompts = df[prompt_column].astype(str).tolist()
    responses = df[answer_column].astype(str).tolist()
    texts = [f"{p}{r}" for p, r in zip(prompts, responses)]

    device = _device()
    print(f"Cache not found. Extracting Self-LSD++ hidden states on {device}.")
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    response_layers, prompt_layers = [], []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting self-LSD representations", unit="batch"):
        batch = texts[start : start + BATCH_SIZE]
        encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        for i in range(hidden.size(0)):
            resp_pos, prompt_pos = _token_span(
                tokenizer,
                prompts[start + i],
                texts[start + i],
                attention_mask[i].cpu(),
                token_window,
                pooling,
            )
            response_layers.append(_pool_layers(hidden[i], resp_pos).cpu())
            prompt_layers.append(_pool_layers(hidden[i], prompt_pos).cpu())

    response_arr = np.stack([x.numpy() for x in response_layers]).astype(np.float32)
    prompt_arr = np.stack([x.numpy() for x in prompt_layers]).astype(np.float32)
    baseline = _baseline_from_response_layers(response_arr)
    payload = {
        "baseline_hidden": baseline,
        "response_layer_vectors": response_arr,
        "prompt_layer_vectors": prompt_arr,
        "raw_label": raw_label,
        "y": y,
        "n_splits": np.asarray([len(splits)], dtype=np.int64),
        "metadata": np.asarray(
            [
                json.dumps(
                    {
                        "model_name": MODEL_NAME,
                        "token_window": token_window,
                        "pooling": pooling,
                        "prompt_column": prompt_column,
                        "answer_column": answer_column,
                        "label_column": label_column,
                        "layer_mapping": "cached index 0 = transformer layer 1",
                    }
                )
            ]
        ),
    }
    for i, (tr, va, te) in enumerate(splits):
        payload[f"split_{i}_train"] = tr
        payload[f"split_{i}_val"] = va
        payload[f"split_{i}_test"] = te
    np.savez_compressed(cache_path, **payload)
    print(f"Saved Self-LSD++ cache: {cache_path}")
    return baseline, response_arr, prompt_arr, raw_label, y, splits


def _safe_cosine_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = np.sum(A * B, axis=-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + eps)
    if not np.isfinite(out).all():
        print("WARNING: non-finite cosine values; replacing with zeros.")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x -= x.mean()
    denom = float(np.sum(x * x))
    return 0.0 if denom <= 1e-12 else float(np.sum(x * (values - values.mean())) / denom)


def _sign_changes(values: np.ndarray) -> int:
    signs = np.sign(values)
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1])) if signs.size >= 2 else 0


def _late_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values[-max(1, int(np.ceil(values.size / 3))) :].mean())


def _normalized_trapezoid_auc(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    area = float(values[1:-1].sum() + 0.5 * (values[0] + values[-1]))
    return area / max(1, values.size - 1)


def _curve_features(prefix: str, base: str, values: np.ndarray) -> tuple[list[float], list[str]]:
    delta = np.diff(values)
    if delta.size == 0:
        delta = np.asarray([0.0])
    vals = [
        values[0],
        values[-1],
        values.mean(),
        values.std(),
        values.min(),
        values.max(),
        np.median(values),
        values.max() - values.min(),
        values[-1] - values[0],
        _slope(values),
        _normalized_trapezoid_auc(values),
        int(np.argmax(values)) / max(1, values.size - 1),
        values.max() - values[-1],
        delta.mean(),
        delta.std(),
        delta.min(),
        delta.max(),
        delta[-1],
        float(np.mean(delta > 0)),
        float(np.mean(delta < 0)),
        _sign_changes(delta),
    ]
    names = [
        f"{prefix}__{base}_first",
        f"{prefix}__{base}_last",
        f"{prefix}__{base}_mean",
        f"{prefix}__{base}_std",
        f"{prefix}__{base}_min",
        f"{prefix}__{base}_max",
        f"{prefix}__{base}_median",
        f"{prefix}__{base}_range",
        f"{prefix}__{base}_gain",
        f"{prefix}__{base}_slope",
        f"{prefix}__{base}_auc",
        f"{prefix}__argmax_{base}_layer_normalized",
        f"{prefix}__post_peak_drop",
        f"{prefix}__delta_{base}_mean",
        f"{prefix}__delta_{base}_std",
        f"{prefix}__delta_{base}_min",
        f"{prefix}__delta_{base}_max",
        f"{prefix}__delta_{base}_last",
        f"{prefix}__fraction_positive_deltas",
        f"{prefix}__fraction_negative_deltas",
        f"{prefix}__num_sign_changes",
    ]
    return [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in vals], names


def _anchor(response_layers: np.ndarray, layer_range: tuple[int, int], mode: str) -> np.ndarray:
    if mode == "late_mean21_24":
        return response_layers[:, 20:24, :].mean(axis=1)
    if mode == "final":
        return response_layers[:, -1, :]
    if mode == "range_end":
        return response_layers[:, layer_range[1] - 1, :]
    if mode == "range_mean":
        return response_layers[:, layer_range[0] - 1 : layer_range[1], :].mean(axis=1)
    raise ValueError("anchor-mode must be late_mean21_24, final, range_end, or range_mean.")


def build_mozeel_lsd_feature_matrix(response_layers: np.ndarray, layer_range: tuple[int, int], anchor_mode: str) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    prefix = f"mozeel_{start}_{end}"
    anchors = _anchor(response_layers, layer_range, anchor_mode)
    rows = []
    for layers, anchor in zip(response_layers, anchors):
        Z = layers[start - 1 : end]
        cos = _safe_cosine_matrix(Z, np.broadcast_to(anchor, Z.shape))
        steps = np.linalg.norm(np.diff(Z, axis=0), axis=1)
        if steps.size == 0:
            steps = np.asarray([0.0])
        rows.append(
            [
                cos[0], cos[-1], cos.mean(), cos.std(), cos.min(), cos.max(), _slope(cos),
                steps.mean(), steps.std(), steps.min(), steps.max(), steps[-1], _slope(steps),
            ]
        )
    names = [
        f"{prefix}__cos_to_anchor_first",
        f"{prefix}__cos_to_anchor_last",
        f"{prefix}__cos_to_anchor_mean",
        f"{prefix}__cos_to_anchor_std",
        f"{prefix}__cos_to_anchor_min",
        f"{prefix}__cos_to_anchor_max",
        f"{prefix}__cos_to_anchor_slope",
        f"{prefix}__step_l2_mean",
        f"{prefix}__step_l2_std",
        f"{prefix}__step_l2_min",
        f"{prefix}__step_l2_max",
        f"{prefix}__step_l2_last",
        f"{prefix}__step_l2_slope",
    ]
    return np.asarray(rows, dtype=np.float32), names


def build_self_alignment_feature_matrix(response_layers: np.ndarray, layer_range: tuple[int, int], anchor_mode: str) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    prefix = f"self_align_{start}_{end}"
    anchors = _anchor(response_layers, layer_range, anchor_mode)
    rows, names = [], None
    for layers, anchor in zip(response_layers, anchors):
        Z = layers[start - 1 : end]
        align = _safe_cosine_matrix(Z, np.broadcast_to(anchor, Z.shape))
        vals, names = _curve_features(prefix, "self_alignment", align)
        rows.append(vals)
    return np.asarray(rows, dtype=np.float32), names or []


def build_directional_anchor_feature_matrix(response_layers: np.ndarray, layer_range: tuple[int, int], anchor_mode: str) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    prefix = f"direction_anchor_{start}_{end}"
    anchors = _anchor(response_layers, layer_range, anchor_mode)
    rows = []
    for layers, anchor in zip(response_layers, anchors):
        Z = layers[start - 1 : end]
        if Z.shape[0] < 2:
            q = np.asarray([0.0])
        else:
            q = _safe_cosine_matrix(Z[1:] - Z[:-1], anchor[None, :] - Z[:-1])
        rows.append(
            [
                q[0], q[-1], q.mean(), q.std(), q.min(), q.max(), np.median(q),
                q.max() - q.min(), _slope(q),
                _normalized_trapezoid_auc(q),
                _late_mean(q), float(np.mean(q > 0)), float(np.mean(q < 0)),
                int(np.sum(q < 0)), _sign_changes(q),
            ]
        )
    names = [
        f"{prefix}__direction_to_anchor_first",
        f"{prefix}__direction_to_anchor_last",
        f"{prefix}__direction_to_anchor_mean",
        f"{prefix}__direction_to_anchor_std",
        f"{prefix}__direction_to_anchor_min",
        f"{prefix}__direction_to_anchor_max",
        f"{prefix}__direction_to_anchor_median",
        f"{prefix}__direction_to_anchor_range",
        f"{prefix}__direction_to_anchor_slope",
        f"{prefix}__direction_to_anchor_auc",
        f"{prefix}__direction_to_anchor_late_mean",
        f"{prefix}__fraction_steps_toward_anchor",
        f"{prefix}__fraction_steps_away_from_anchor",
        f"{prefix}__num_negative_anchor_steps",
        f"{prefix}__num_sign_changes_direction",
    ]
    return np.asarray(rows, dtype=np.float32), names


def build_anchor_progress_feature_matrix(response_layers: np.ndarray, layer_range: tuple[int, int], anchor_mode: str) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    prefix = f"progress_anchor_{start}_{end}"
    anchors = _anchor(response_layers, layer_range, anchor_mode)
    rows = []
    for layers, anchor in zip(response_layers, anchors):
        d = np.linalg.norm(layers[start - 1 : end] - anchor[None, :], axis=1)
        drop = d[0] - d[-1]
        rows.append(
            [
                d[0], d[-1], d.mean(), d.std(), d.min(), d.max(), d.max() - d.min(),
                drop, drop / (abs(d[0]) + 1e-8), _slope(d),
                int(np.argmin(d)) / max(1, d.size - 1), _late_mean(d),
            ]
        )
    names = [
        f"{prefix}__distance_to_anchor_first",
        f"{prefix}__distance_to_anchor_last",
        f"{prefix}__distance_to_anchor_mean",
        f"{prefix}__distance_to_anchor_std",
        f"{prefix}__distance_to_anchor_min",
        f"{prefix}__distance_to_anchor_max",
        f"{prefix}__distance_to_anchor_range",
        f"{prefix}__distance_to_anchor_drop",
        f"{prefix}__distance_to_anchor_drop_ratio",
        f"{prefix}__distance_to_anchor_slope",
        f"{prefix}__argmin_distance_layer_normalized",
        f"{prefix}__late_distance_mean",
    ]
    return np.asarray(rows, dtype=np.float32), names


def build_orthogonal_drift_feature_matrix(response_layers: np.ndarray, layer_range: tuple[int, int], anchor_mode: str) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    prefix = f"orthogonal_drift_{start}_{end}"
    anchors = _anchor(response_layers, layer_range, anchor_mode)
    rows = []
    for layers, anchor in zip(response_layers, anchors):
        Z = layers[start - 1 : end]
        if Z.shape[0] < 2:
            toward = orth = step = np.zeros(1)
        else:
            v = Z[1:] - Z[:-1]
            d = anchor[None, :] - Z[:-1]
            d_unit = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
            toward = np.sum(v * d_unit, axis=1)
            toward_vec = toward[:, None] * d_unit
            orth_vec = v - toward_vec
            orth = np.linalg.norm(orth_vec, axis=1)
            step = np.linalg.norm(v, axis=1)
        ratio = orth / (step + 1e-8)
        orth_to_toward = orth / (np.abs(toward) + 1e-8)
        rows.append(
            [
                toward.mean(), toward.std(), toward.min(), toward.max(), _late_mean(toward),
                orth.mean(), orth.std(), orth.max(), ratio.mean(), ratio.max(),
                orth_to_toward.mean(), orth_to_toward.max(), float(np.mean(toward > 0)),
            ]
        )
    names = [
        f"{prefix}__toward_component_mean",
        f"{prefix}__toward_component_std",
        f"{prefix}__toward_component_min",
        f"{prefix}__toward_component_max",
        f"{prefix}__toward_component_late_mean",
        f"{prefix}__orthogonal_drift_mean",
        f"{prefix}__orthogonal_drift_std",
        f"{prefix}__orthogonal_drift_max",
        f"{prefix}__orthogonal_ratio_mean",
        f"{prefix}__orthogonal_ratio_max",
        f"{prefix}__orthogonal_to_toward_ratio_mean",
        f"{prefix}__orthogonal_to_toward_ratio_max",
        f"{prefix}__fraction_toward_positive",
    ]
    return np.asarray(rows, dtype=np.float32), names


def build_prompt_response_coupling_feature_matrix(prompt_layers: np.ndarray, response_layers: np.ndarray, layer_range: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    prefix = f"prompt_response_{start}_{end}"
    rows = []
    for P, H in zip(prompt_layers, response_layers):
        Pz, Hz = P[start - 1 : end], H[start - 1 : end]
        c = _safe_cosine_matrix(Pz, Hz)
        diff_norm = np.linalg.norm(Hz - Pz, axis=1)
        rows.append(
            [
                c[0], c[-1], c.mean(), c.std(), c.min(), c.max(), c.max() - c.min(),
                c[-1] - c[0], _slope(c), _late_mean(c),
                diff_norm.mean(), diff_norm.std(), diff_norm[-1], _slope(diff_norm),
            ]
        )
    names = [
        f"{prefix}__prompt_response_cos_first",
        f"{prefix}__prompt_response_cos_last",
        f"{prefix}__prompt_response_cos_mean",
        f"{prefix}__prompt_response_cos_std",
        f"{prefix}__prompt_response_cos_min",
        f"{prefix}__prompt_response_cos_max",
        f"{prefix}__prompt_response_cos_range",
        f"{prefix}__prompt_response_cos_gain",
        f"{prefix}__prompt_response_cos_slope",
        f"{prefix}__prompt_response_cos_late_mean",
        f"{prefix}__response_minus_prompt_norm_mean",
        f"{prefix}__response_minus_prompt_norm_std",
        f"{prefix}__response_minus_prompt_norm_last",
        f"{prefix}__response_minus_prompt_norm_slope",
    ]
    return np.asarray(rows, dtype=np.float32), names


class PostPcaFeatureBlockBuilder:
    def __init__(self, pca_components: int, block_names: list[str]) -> None:
        self.pca_components = pca_components
        self.block_names = block_names
        self.hidden_scaler = StandardScaler()
        self.pca: PCA | None = None
        self.scalers = {name: StandardScaler() for name in block_names}

    def fit_transform(self, X_hidden: np.ndarray | None, blocks: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        if X_hidden is not None:
            hidden_scaled = self.hidden_scaler.fit_transform(X_hidden)
            n = min(self.pca_components, hidden_scaled.shape[0] - 1, hidden_scaled.shape[1])
            self.pca = PCA(n_components=n, random_state=42)
            parts.append(self.pca.fit_transform(hidden_scaled))
        for name in self.block_names:
            parts.append(self.scalers[name].fit_transform(blocks[name]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform(self, X_hidden: np.ndarray | None, blocks: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        if X_hidden is not None:
            if self.pca is None:
                raise RuntimeError("PCA is not fitted.")
            parts.append(self.pca.transform(self.hidden_scaler.transform(X_hidden)))
        for name in self.block_names:
            parts.append(self.scalers[name].transform(blocks[name]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def hidden_pca_dim(self) -> int:
        return int(self.pca.n_components_) if self.pca is not None else 0


def evaluate_scores_with_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(auc),
    }


def find_best_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray, num_thresholds: int) -> tuple[float, dict]:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        candidates = np.asarray([0.0])
    elif np.unique(finite).size == 1:
        candidates = np.asarray([float(finite[0]), 0.0])
    else:
        candidates = np.unique(np.concatenate([np.linspace(finite.min(), finite.max(), num_thresholds), [0.0]]))
    best_t, best_m, best_key = 0.0, {}, (-1.0, -1.0, -1.0)
    for t in candidates:
        m = evaluate_scores_with_threshold(y_true, scores, float(t))
        key = (m["f1"], m["precision"], m["accuracy"])
        if key > best_key:
            best_t, best_m, best_key = float(t), m, key
    return best_t, best_m


def _feature_group(name: str) -> str:
    if name.startswith("pca_"):
        return "hidden_pca"
    for group in BLOCKS:
        prefix = {
            "trajectory": "traj_",
            "mozeel_lsd": "mozeel_",
            "self_alignment": "self_align_",
            "directional_anchor": "direction_anchor_",
            "progress_anchor": "progress_anchor_",
            "orthogonal_drift": "orthogonal_drift_",
            "prompt_response_coupling": "prompt_response_",
        }[group]
        if name.startswith(prefix):
            return group
    return "other"


def _aggregate(rows: list[dict]) -> dict:
    out = {}
    for split in ("val", "test"):
        for mode in ("default", "tuned"):
            for metric in ("accuracy", "precision", "recall", "f1"):
                key = f"{split}_{metric}_{mode}"
                out[key] = float(np.mean([row[key] for row in rows]))
        out[f"{split}_roc_auc"] = float(np.mean([row[f"{split}_roc_auc"] for row in rows]))
    for key in ("best_threshold", "final_feature_dim", "hidden_pca_dim"):
        out[key] = float(np.mean([row[key] for row in rows]))
    return out


def _prediction_frame(fold: int, split: str, indices: np.ndarray, raw_label: np.ndarray, y: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": fold,
            "sample_id": indices,
            "y_true": y[indices],
            "raw_label": raw_label[indices],
            "decision_score": scores,
            "y_pred_default": (scores >= 0.0).astype(int),
            "y_pred_tuned": (scores >= threshold).astype(int),
            "threshold_default": 0.0,
            "threshold_tuned": threshold,
            "split": split,
        }
    )


def evaluate_experiment(
    experiment: str,
    layer_range: str,
    X_hidden: np.ndarray,
    blocks: dict[str, np.ndarray],
    block_names: dict[str, list[str]],
    enabled: list[str],
    raw_label: np.ndarray,
    y: np.ndarray,
    splits: list,
    pca_components: int,
    svm_c: float,
    seed: int,
    num_thresholds: int,
    predictions_dir: Path,
    use_hidden: bool = True,
) -> tuple[dict, list[str], tuple[SVC, np.ndarray, np.ndarray, float] | None]:
    rows, pred_frames, feature_names, artifact = [], [], [], None
    t0 = time.time()
    for fold, (tr, va, te) in enumerate(splits, 1):
        if va is None:
            raise ValueError("Validation split required.")
        builder = PostPcaFeatureBlockBuilder(pca_components, enabled)
        train_blocks = {name: blocks[name][tr] for name in enabled}
        val_blocks = {name: blocks[name][va] for name in enabled}
        test_blocks = {name: blocks[name][te] for name in enabled}
        X_train = builder.fit_transform(X_hidden[tr] if use_hidden else None, train_blocks)
        X_val = builder.transform(X_hidden[va] if use_hidden else None, val_blocks)
        X_test = builder.transform(X_hidden[te] if use_hidden else None, test_blocks)
        names = [f"pca_{i}" for i in range(builder.hidden_pca_dim)]
        for name in enabled:
            names.extend(block_names[name])
        if not feature_names:
            feature_names = names
        model = SVC(C=svm_c, kernel="linear", probability=False, random_state=seed)
        model.fit(X_train, y[tr])
        val_scores = model.decision_function(X_val)
        test_scores = model.decision_function(X_test)
        best_t, val_tuned = find_best_threshold_by_f1(y[va], val_scores, num_thresholds)
        val_default = evaluate_scores_with_threshold(y[va], val_scores, 0.0)
        test_default = evaluate_scores_with_threshold(y[te], test_scores, 0.0)
        test_tuned = evaluate_scores_with_threshold(y[te], test_scores, best_t)
        row = {
            "fold": fold,
            "best_threshold": best_t,
            "final_feature_dim": X_train.shape[1],
            "hidden_pca_dim": builder.hidden_pca_dim,
            "val_roc_auc": val_default["roc_auc"],
            "test_roc_auc": test_default["roc_auc"],
        }
        for split, metrics in (("val", val_default), ("test", test_default)):
            for metric in ("accuracy", "precision", "recall", "f1"):
                row[f"{split}_{metric}_default"] = metrics[metric]
        for split, metrics in (("val", val_tuned), ("test", test_tuned)):
            for metric in ("accuracy", "precision", "recall", "f1"):
                row[f"{split}_{metric}_tuned"] = metrics[metric]
        rows.append(row)
        pred_frames.extend(
            [
                _prediction_frame(fold, "val", va, raw_label, y, val_scores, best_t),
                _prediction_frame(fold, "test", te, raw_label, y, test_scores, best_t),
            ]
        )
        artifact = (model, X_val, y[va], best_t)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_frames, ignore_index=True).to_csv(predictions_dir / f"{experiment}.csv", index=False)
    summary = {
        "experiment": experiment,
        "layer_range": layer_range,
        "anchor_mode": "",
        "pooling": "",
        "use_hidden": use_hidden,
        "pca_components": pca_components if use_hidden else 0,
        "svm_c": svm_c,
        "runtime_seconds": time.time() - t0,
        **{f"use_{block}": block in enabled for block in BLOCKS},
        **{f"{block}_feature_dim": blocks[block].shape[1] if block in enabled else 0 for block in BLOCKS},
        **_aggregate(rows),
    }
    return summary, feature_names, artifact


def permutation_importance_with_threshold(model: SVC, X: np.ndarray, y: np.ndarray, threshold: float, feature_names: list[str], n_repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_scores = model.decision_function(X)
    base_f1 = float(f1_score(y, (base_scores >= threshold).astype(int), zero_division=0))
    rows = []
    for idx, name in enumerate(feature_names):
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            order = rng.permutation(Xp.shape[0])
            Xp[:, idx] = Xp[order, idx]
            scores = model.decision_function(Xp)
            vals.append(float(f1_score(y, (scores >= threshold).astype(int), zero_division=0)))
        arr = np.asarray(vals)
        rows.append(
            {
                "feature_name": name,
                "feature_index": idx,
                "importance_mean": float(base_f1 - arr.mean()),
                "importance_std": float(arr.std()),
                "base_f1": base_f1,
                "permuted_f1_mean": float(arr.mean()),
                "permuted_f1_std": float(arr.std()),
                "feature_group": _feature_group(name),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def _debug_subset(X_hidden, response, prompt, raw_label, y, splits, n):
    if n is None or n >= len(y):
        return X_hidden, response, prompt, raw_label, y, splits
    keep = set(range(n))
    remap = {old: new for new, old in enumerate(sorted(keep))}
    new_splits = []
    for tr, va, te in splits:
        trn = np.asarray([remap[int(i)] for i in tr if int(i) in keep], dtype=int)
        van = np.asarray([remap[int(i)] for i in va if int(i) in keep], dtype=int) if va is not None else None
        ten = np.asarray([remap[int(i)] for i in te if int(i) in keep], dtype=int)
        if trn.size and van is not None and van.size and ten.size:
            new_splits.append((trn, van, ten))
    if not new_splits:
        raise ValueError("--max-samples-debug removed all usable folds.")
    return X_hidden[:n], response[:n], prompt[:n], raw_label[:n], y[:n], new_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "self_lsd_plus_ablation")
    parser.add_argument("--layer-ranges", nargs="+", default=["10-24", "12-20", "21-24", "1-24"])
    parser.add_argument("--token-window", type=int, default=32)
    parser.add_argument("--num-thresholds", type=int, default=1000)
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--answer-column", default="response")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--truthful-label", type=int, default=0)
    parser.add_argument("--hallucinated-label", type=int, default=1)
    parser.add_argument("--anchor-mode", default="late_mean21_24")
    parser.add_argument("--pooling", default="response_last_k_mean", choices=["response_last_k_mean", "response_last_token", "full_last_k_mean"])
    parser.add_argument("--run-permutation-importance", action="store_true")
    parser.add_argument("--importance-experiment", default=None)
    parser.add_argument("--importance-repeats", type=int, default=20)
    parser.add_argument("--max-samples-debug", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = args.output_dir
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "positive_class": "hallucinated",
            "model_name": MODEL_NAME,
            "layer_mapping": "cached index 0 = transformer layer 1; hidden_states[1] = transformer layer 1 when embeddings are present",
            "note": "Reference-free Self-LSD++: own-response late-layer anchor, not truth/reference alignment.",
        }
    )
    with (output_dir / "config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2)

    X_hidden, response_layers, prompt_layers, raw_label, y, splits = load_or_extract_self_lsd_cache(
        output_dir,
        args.token_window,
        args.pooling,
        args.prompt_column,
        args.answer_column,
        args.label_column,
        args.truthful_label,
        args.hallucinated_label,
    )
    X_hidden, response_layers, prompt_layers, raw_label, y, splits = _debug_subset(
        X_hidden, response_layers, prompt_layers, raw_label, y, splits, args.max_samples_debug
    )
    print(f"Samples: {len(y)}")
    print(f"Baseline hidden shape: {X_hidden.shape}")
    print(f"Response layer vectors: {response_layers.shape}")
    print(f"Prompt layer vectors: {prompt_layers.shape}")
    for i, (tr, va, te) in enumerate(splits, 1):
        print(f"Fold {i}: train={len(tr)} {_class_distribution(y, tr)} val={len(va)} {_class_distribution(y, va)} test={len(te)} {_class_distribution(y, te)}")

    block_mats: dict[str, dict[str, np.ndarray]] = {}
    block_names: dict[str, dict[str, list[str]]] = {}
    for range_label in args.layer_ranges:
        layer_range = _parse_range(range_label)
        mats, names = {}, {}
        mats["trajectory"], names["trajectory"] = build_trajectory_feature_matrix(response_layers, layer_range)
        mats["mozeel_lsd"], names["mozeel_lsd"] = build_mozeel_lsd_feature_matrix(response_layers, layer_range, args.anchor_mode)
        mats["self_alignment"], names["self_alignment"] = build_self_alignment_feature_matrix(response_layers, layer_range, args.anchor_mode)
        mats["directional_anchor"], names["directional_anchor"] = build_directional_anchor_feature_matrix(response_layers, layer_range, args.anchor_mode)
        mats["progress_anchor"], names["progress_anchor"] = build_anchor_progress_feature_matrix(response_layers, layer_range, args.anchor_mode)
        mats["orthogonal_drift"], names["orthogonal_drift"] = build_orthogonal_drift_feature_matrix(response_layers, layer_range, args.anchor_mode)
        mats["prompt_response_coupling"], names["prompt_response_coupling"] = build_prompt_response_coupling_feature_matrix(prompt_layers, response_layers, layer_range)
        for key in mats:
            if not np.isfinite(mats[key]).all():
                print(f"WARNING: non-finite values in {key} {range_label}; replacing with zeros.")
                mats[key] = np.nan_to_num(mats[key], nan=0.0, posinf=0.0, neginf=0.0)
        block_mats[range_label], block_names[range_label] = mats, names
        print(f"Prepared range {range_label}: " + ", ".join(f"{k}={v.shape}" for k, v in mats.items()))

    specs = [("E0_baseline_only", "none", [], True)]
    for range_label in args.layer_ranges:
        safe = range_label.replace("-", "_")
        specs.extend(
            [
                (f"E1_current_trajectory_{safe}", range_label, ["trajectory"], True),
                (f"E2_mozeel_lsd_{safe}", range_label, ["mozeel_lsd"], True),
                (f"E3_self_alignment_{safe}", range_label, ["self_alignment"], True),
                (f"E4_directional_anchor_{safe}", range_label, ["directional_anchor"], True),
                (f"E5_progress_anchor_{safe}", range_label, ["progress_anchor"], True),
                (f"E6_orthogonal_drift_{safe}", range_label, ["orthogonal_drift"], True),
                (f"E7_self_lsd_plus_{safe}", range_label, ["self_alignment", "directional_anchor", "progress_anchor", "orthogonal_drift"], True),
                (f"E8_prompt_response_coupling_{safe}", range_label, ["prompt_response_coupling"], True),
                (f"E9_self_lsd_plus_prompt_coupling_{safe}", range_label, ["self_alignment", "directional_anchor", "progress_anchor", "orthogonal_drift", "prompt_response_coupling"], True),
                (f"E10_trajectory_plus_self_lsd_plus_prompt_coupling_{safe}", range_label, list(BLOCKS), True),
            ]
        )

    empty = {name: np.zeros((len(y), 0), dtype=np.float32) for name in BLOCKS}
    empty_names = {name: [] for name in BLOCKS}
    summaries, feature_names_by_exp, artifacts = [], {}, {}
    for exp, range_label, enabled, use_hidden in specs:
        mats = empty if range_label == "none" else block_mats[range_label]
        names = empty_names if range_label == "none" else block_names[range_label]
        print(f"\n{exp}: range={range_label}, blocks={enabled}")
        summary, feature_names, artifact = evaluate_experiment(
            exp, range_label, X_hidden, mats, names, enabled, raw_label, y, splits,
            args.pca_components, args.svm_c, args.seed, args.num_thresholds,
            predictions_dir, use_hidden=use_hidden,
        )
        summary["anchor_mode"] = args.anchor_mode
        summary["pooling"] = args.pooling
        summaries.append(summary)
        feature_names_by_exp[exp] = {
            "all": feature_names,
            **({name: names[name] for name in enabled} if range_label != "none" else {}),
            "hidden_pca": [name for name in feature_names if name.startswith("pca_")],
        }
        artifacts[exp] = (artifact, feature_names)
        print(f"  val_f1_tuned={summary['val_f1_tuned']:.4f} test_f1_tuned={summary['test_f1_tuned']:.4f} test_auc={summary['test_roc_auc']:.4f}")
        pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    summary_df.to_json(output_dir / "summary.json", orient="records", indent=2)
    with (output_dir / "feature_names.json").open("w") as f:
        json.dump(feature_names_by_exp, f, indent=2)

    if args.run_permutation_importance:
        imp_exp = args.importance_experiment
        if imp_exp is None:
            imp_exp = str(
                summary_df.loc[summary_df["experiment"] != "E0_baseline_only"]
                .sort_values("val_f1_tuned", ascending=False)
                .iloc[0]["experiment"]
            )
        print(f"\nPermutation importance experiment: {imp_exp}")
        artifact, feature_names = artifacts[imp_exp]
        model, X_val, y_val, threshold = artifact
        imp = permutation_importance_with_threshold(model, X_val, y_val, threshold, feature_names, args.importance_repeats, args.seed)
        imp.to_csv(output_dir / "permutation_importance_best_val.csv", index=False)
        imp.loc[imp["feature_group"] != "hidden_pca"].to_csv(output_dir / "permutation_importance_nonhidden_only.csv", index=False)
        imp.loc[imp["feature_group"].isin(["self_alignment", "directional_anchor", "progress_anchor", "orthogonal_drift"])].to_csv(
            output_dir / "permutation_importance_self_lsd_plus_only.csv",
            index=False,
        )
        imp.loc[imp["feature_group"] == "prompt_response_coupling"].to_csv(
            output_dir / "permutation_importance_prompt_response_coupling_only.csv",
            index=False,
        )

    leaderboard_cols = [
        "experiment",
        "layer_range",
        "final_feature_dim",
        "val_f1_tuned",
        "test_f1_tuned",
        "val_roc_auc",
        "test_roc_auc",
        "best_threshold",
    ]
    print("\nSelf-LSD++ leaderboard")
    print(summary_df.sort_values("val_f1_tuned", ascending=False)[leaderboard_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best_val = summary_df.sort_values("val_f1_tuned", ascending=False).iloc[0]
    best_test = summary_df.sort_values("test_f1_tuned", ascending=False).iloc[0]
    e0 = summary_df.loc[summary_df["experiment"] == "E0_baseline_only"].iloc[0]
    print(f"\nBest by val_f1_tuned: {best_val['experiment']} ({best_val['val_f1_tuned']:.4f})")
    print(f"Best by test_f1_tuned: {best_test['experiment']} ({best_test['test_f1_tuned']:.4f})")
    print(
        "Delta vs E0_baseline_only: "
        f"val_f1_tuned_delta={best_val['val_f1_tuned'] - e0['val_f1_tuned']:.4f}, "
        f"test_f1_tuned_delta={best_val['test_f1_tuned'] - e0['test_f1_tuned']:.4f}, "
        f"test_roc_auc_delta={best_val['test_roc_auc'] - e0['test_roc_auc']:.4f}"
    )
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
