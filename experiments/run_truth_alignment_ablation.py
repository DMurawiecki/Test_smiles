"""
Mini-LSD-style truth-alignment ablation.

This is a lightweight reference-conditioned experiment: no projection head, no
contrastive loss.  It keeps the existing split protocol, uses mean(21-24)
baseline hidden features with train-only scaler/PCA, appends separately scaled
feature blocks, and evaluates a linear SVM with validation-tuned thresholds.

Layer convention: cached layer-vector index 0 corresponds to transformer layer
1.  When model outputs include embeddings, hidden_states[1] is transformer
layer 1 and hidden_states[24] is transformer layer 24.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    BASELINE_AGGREGATION_MODE,
    BASELINE_SPECTRAL_MODE,
    DATA_FILE,
    N_TRANSFORMER_LAYERS,
    _class_distribution,
    _device,
    _parse_range,
    build_trajectory_feature_matrix,
    load_or_extract_representations,
)
from model import MAX_LENGTH, get_model_and_tokenizer  # noqa: E402


BATCH_SIZE = 4
PROMPT_CANDIDATES = ("question", "prompt", "input", "query", "problem")
ANSWER_CANDIDATES = (
    "answer",
    "model_answer",
    "generated_answer",
    "prediction",
    "response",
    "output",
)
REFERENCE_CANDIDATES = (
    "reference_answer",
    "gold_answer",
    "correct_answer",
    "target",
    "ground_truth",
    "reference",
    "gold",
    "label_answer",
    "expected_answer",
)
LABEL_CANDIDATES = ("label", "labels", "target_label", "y")


@dataclass
class FoldArtifact:
    fold: int
    model: SVC
    X_val: np.ndarray
    y_val: np.ndarray
    threshold: float
    feature_names: list[str]


def _detect_column(df: pd.DataFrame, provided: str | None, candidates: tuple[str, ...], role: str) -> str:
    if provided is not None:
        if provided not in df.columns:
            raise ValueError(f"{role} column {provided!r} not found. Available: {df.columns.tolist()}")
        return provided
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not auto-detect {role} column. Available: {df.columns.tolist()}")


def _safe_cosine_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    numer = np.sum(A * B, axis=-1)
    denom = np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1)
    return numer / (denom + eps)


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * (values - values.mean())) / denom)


def _num_sign_changes(values: np.ndarray) -> int:
    if values.size < 2:
        return 0
    signs = np.sign(values)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _safe_matrix(X: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(X).all():
        print(f"WARNING: non-finite values found in {name}; replacing with zeros.")
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _reference_cache_file(output_dir: Path, token_window: int) -> Path:
    return output_dir / f"truth_reference_layer_vectors_tail{token_window}.npz"


def _answer_span_positions(tokenizer, prompt: str, full_text: str, seq_len: int, token_window: int) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    start = min(len(prompt_ids), max(len(full_ids) - 1, 0), seq_len - 1)
    end = min(len(full_ids), seq_len)
    if end <= start:
        start = max(0, min(seq_len - 1, end - 1))
        end = min(seq_len, start + 1)
    positions = torch.arange(start, end, dtype=torch.long)
    if positions.numel() == 0:
        positions = torch.tensor([seq_len - 1], dtype=torch.long)
    return positions[-min(token_window, positions.numel()) :]


def _pool_layer_span(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    pooled = []
    hs = aggregation._as_layer_tensor(hidden_states).detach()
    for layer_number in range(1, N_TRANSFORMER_LAYERS + 1):
        idx = aggregation.transformer_layer_to_index(layer_number, hs)
        layer = hs[idx].index_select(0, positions.to(hs.device)).to(dtype=torch.float32)
        pooled.append(layer.mean(dim=0))
    return torch.stack(pooled, dim=0)


def load_or_extract_truth_layer_vectors(
    output_dir: Path,
    token_window: int,
    prompt_column: str | None,
    reference_column: str | None,
    label_column: str | None,
    max_samples_debug: int | None = None,
) -> np.ndarray:
    """Return reference-answer pooled vectors, shape [N, 24, hidden_dim]."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _reference_cache_file(output_dir, token_window)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        vectors = np.asarray(data["truth_layer_vectors"], dtype=np.float32)
        return vectors[:max_samples_debug] if max_samples_debug is not None else vectors

    df = pd.read_csv(DATA_FILE)
    prompt_col = _detect_column(df, prompt_column, PROMPT_CANDIDATES, "prompt/question")
    reference_col = _detect_column(df, reference_column, REFERENCE_CANDIDATES, "reference/gold answer")
    _detect_column(df, label_column, LABEL_CANDIDATES, "label")

    device = _device()
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    prompts = df[prompt_col].astype(str).tolist()
    refs = df[reference_col].astype(str).tolist()
    full_texts = [f"{prompt}{ref}" for prompt, ref in zip(prompts, refs)]
    vectors: list[torch.Tensor] = []

    for start in tqdm(range(0, len(full_texts), BATCH_SIZE), desc="Extracting truth vectors", unit="batch"):
        batch_texts = full_texts[start : start + BATCH_SIZE]
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
        for i in range(hidden.size(0)):
            seq_len = int(attention_mask[i].sum().item())
            positions = _answer_span_positions(
                tokenizer,
                prompts[start + i],
                full_texts[start + i],
                seq_len,
                token_window,
            )
            vectors.append(_pool_layer_span(hidden[i], positions).cpu())

    truth_layer_vectors = np.stack([item.numpy() for item in vectors]).astype(np.float32)
    np.savez_compressed(cache_path, truth_layer_vectors=truth_layer_vectors)
    print(f"Saved truth reference cache: {cache_path}")
    return truth_layer_vectors


def truth_vectors_from_layers(
    truth_layer_vectors: np.ndarray,
    mode: str,
    layer_range: tuple[int, int],
) -> np.ndarray:
    if mode == "mean21_24":
        start, end = 21, 24
    elif mode == "last":
        return truth_layer_vectors[:, 23, :]
    elif mode == "same_range_mean":
        start, end = layer_range
    else:
        raise ValueError("truth_vector_mode must be mean21_24, last, or same_range_mean.")
    return truth_layer_vectors[:, start - 1 : end, :].mean(axis=1)


def build_alignment_feature_matrix(
    layer_vectors: np.ndarray,
    truth_vectors: np.ndarray,
    layer_range: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    prefix = f"align_{start}_{end}__"
    names = [
        "alignment_first",
        "alignment_last",
        "alignment_mean",
        "alignment_std",
        "alignment_min",
        "alignment_max",
        "alignment_median",
        "alignment_range",
        "alignment_gain",
        "alignment_slope",
        "alignment_auc",
        "argmax_alignment_layer_normalized",
        "post_peak_drop",
        "delta_alignment_mean",
        "delta_alignment_std",
        "delta_alignment_min",
        "delta_alignment_max",
        "delta_alignment_last",
        "fraction_positive_deltas",
        "fraction_negative_deltas",
        "num_sign_changes",
    ]
    rows = []
    for layers, truth in zip(layer_vectors, truth_vectors):
        Z = layers[start - 1 : end]
        truth_rep = np.broadcast_to(truth, Z.shape)
        align = _safe_cosine_matrix(Z, truth_rep)
        delta = np.diff(align)
        if delta.size == 0:
            delta = np.asarray([0.0])
        row = [
            align[0],
            align[-1],
            align.mean(),
            align.std(),
            align.min(),
            align.max(),
            np.median(align),
            align.max() - align.min(),
            align[-1] - align[0],
            _slope(align),
            np.trapz(align) / max(1, align.size - 1) if align.size > 1 else align.mean(),
            int(np.argmax(align)) / max(1, align.size - 1),
            align.max() - align[-1],
            delta.mean(),
            delta.std(),
            delta.min(),
            delta.max(),
            delta[-1],
            float(np.mean(delta > 0)),
            float(np.mean(delta < 0)),
            _num_sign_changes(delta),
        ]
        rows.append(row)
    X = _safe_matrix(np.asarray(rows, dtype=np.float32), f"alignment {start}-{end}")
    return X, [prefix + name for name in names]


def build_directional_truth_feature_matrix(
    layer_vectors: np.ndarray,
    truth_vectors: np.ndarray,
    layer_range: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    start, end = layer_range
    prefix = f"truthdir_{start}_{end}__"
    names = [
        "direction_to_truth_first",
        "direction_to_truth_last",
        "direction_to_truth_mean",
        "direction_to_truth_std",
        "direction_to_truth_min",
        "direction_to_truth_max",
        "direction_to_truth_median",
        "direction_to_truth_range",
        "direction_to_truth_slope",
        "direction_to_truth_auc",
        "direction_to_truth_late_mean",
        "fraction_steps_toward_truth",
        "fraction_steps_away_from_truth",
        "num_negative_truth_steps",
        "num_sign_changes_direction",
    ]
    rows = []
    for layers, truth in zip(layer_vectors, truth_vectors):
        Z = layers[start - 1 : end]
        if Z.shape[0] < 2:
            q = np.zeros(1, dtype=np.float64)
        else:
            v = Z[1:] - Z[:-1]
            d = truth[None, :] - Z[:-1]
            q = _safe_cosine_matrix(v, d)
        late = q[-max(1, int(np.ceil(q.size / 3))) :]
        row = [
            q[0],
            q[-1],
            q.mean(),
            q.std(),
            q.min(),
            q.max(),
            np.median(q),
            q.max() - q.min(),
            _slope(q),
            np.trapz(q) / max(1, q.size - 1) if q.size > 1 else q.mean(),
            late.mean(),
            float(np.mean(q > 0)),
            float(np.mean(q < 0)),
            int(np.sum(q < 0)),
            _num_sign_changes(q),
        ]
        rows.append(row)
    X = _safe_matrix(np.asarray(rows, dtype=np.float32), f"directional truth {start}-{end}")
    return X, [prefix + name for name in names]


def evaluate_scores_with_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
    }


def find_best_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray, num_thresholds: int = 1000) -> tuple[float, dict]:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        candidates = np.asarray([0.0])
    elif np.unique(finite).size == 1:
        candidates = np.asarray([float(finite[0]), 0.0])
    else:
        candidates = np.unique(np.concatenate([np.linspace(finite.min(), finite.max(), num_thresholds), [0.0]]))
    best_t, best_metrics, best_key = 0.0, {}, (-1.0, -1.0, -1.0)
    for t in candidates:
        metrics = evaluate_scores_with_threshold(y_true, scores, float(t))
        key = (metrics["f1"], metrics["precision"], metrics["accuracy"])
        if key > best_key:
            best_t, best_metrics, best_key = float(t), metrics, key
    return best_t, best_metrics


class PostPcaFeatureBlockBuilder:
    """Hidden scaler/PCA plus independently scaled non-hidden feature blocks."""

    def __init__(self, pca_components: int, block_names: list[str]) -> None:
        self.pca_components = pca_components
        self.block_names = block_names
        self.hidden_scaler = StandardScaler()
        self.pca: PCA | None = None
        self.block_scalers = {name: StandardScaler() for name in block_names}

    def fit_transform(self, X_hidden: np.ndarray | None, blocks: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        if X_hidden is not None:
            hidden_scaled = self.hidden_scaler.fit_transform(X_hidden)
            n = min(self.pca_components, hidden_scaled.shape[0] - 1, hidden_scaled.shape[1])
            self.pca = PCA(n_components=n, random_state=42)
            parts.append(self.pca.fit_transform(hidden_scaled))
        for name in self.block_names:
            parts.append(self.block_scalers[name].fit_transform(blocks[name]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform(self, X_hidden: np.ndarray | None, blocks: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        if X_hidden is not None:
            if self.pca is None:
                raise RuntimeError("PCA has not been fitted.")
            parts.append(self.pca.transform(self.hidden_scaler.transform(X_hidden)))
        for name in self.block_names:
            parts.append(self.block_scalers[name].transform(blocks[name]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def hidden_pca_dim(self) -> int:
        return int(self.pca.n_components_) if self.pca is not None else 0


def _aggregate(rows: list[dict]) -> dict:
    out = {}
    for split in ("val", "test"):
        for mode in ("default", "tuned"):
            for metric in ("accuracy", "precision", "recall", "f1"):
                key = f"{split}_{metric}_{mode}"
                out[key] = float(np.mean([row[key] for row in rows]))
        out[f"{split}_roc_auc"] = float(np.mean([row[f"{split}_roc_auc"] for row in rows]))
    out["best_threshold"] = float(np.mean([row["best_threshold"] for row in rows]))
    out["final_feature_dim"] = float(np.mean([row["final_feature_dim"] for row in rows]))
    out["hidden_pca_dim"] = float(np.mean([row["hidden_pca_dim"] for row in rows]))
    return out


def _feature_group_names(hidden_dim: int, block_feature_names: dict[str, list[str]], block_order: list[str]) -> list[str]:
    names = [f"pca_{i}" for i in range(hidden_dim)]
    for block in block_order:
        names.extend(block_feature_names[block])
    return names


def _feature_group(name: str) -> str:
    if name.startswith("pca_"):
        return "hidden_pca"
    if name.startswith("traj_"):
        return "trajectory"
    if name.startswith("align_"):
        return "alignment"
    if name.startswith("truthdir_"):
        return "directional_truth"
    return "other"


def permutation_importance_with_threshold(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    feature_names: list[str],
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base_scores = model.decision_function(X)
    base_f1 = float(f1_score(y, (base_scores >= threshold).astype(int), zero_division=0))
    rows = []
    for idx, name in enumerate(feature_names):
        values = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            order = rng.permutation(X_perm.shape[0])
            X_perm[:, idx] = X_perm[order, idx]
            scores = model.decision_function(X_perm)
            values.append(float(f1_score(y, (scores >= threshold).astype(int), zero_division=0)))
        arr = np.asarray(values)
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


def _prediction_frame(fold: int, split: str, indices: np.ndarray, y: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": fold,
            "sample_id": indices,
            "y_true": y[indices],
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
    alignment_range: str,
    X_hidden: np.ndarray,
    feature_blocks: dict[str, np.ndarray],
    block_feature_names: dict[str, list[str]],
    enabled_blocks: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
    pca_components: int,
    svm_c: float,
    seed: int,
    num_thresholds: int,
    predictions_dir: Path,
    use_hidden: bool = True,
) -> tuple[dict, list[dict], list[str], tuple[SVC, np.ndarray, np.ndarray, float] | None]:
    rows, pred_frames = [], []
    artifact = None
    feature_names: list[str] = []
    t0 = time.time()
    for fold, (tr, va, te) in enumerate(splits, 1):
        if va is None:
            raise ValueError("Validation split is required for threshold tuning.")
        builder = PostPcaFeatureBlockBuilder(pca_components, enabled_blocks)
        train_blocks = {name: feature_blocks[name][tr] for name in enabled_blocks}
        val_blocks = {name: feature_blocks[name][va] for name in enabled_blocks}
        test_blocks = {name: feature_blocks[name][te] for name in enabled_blocks}
        X_train = builder.fit_transform(X_hidden[tr] if use_hidden else None, train_blocks)
        X_val = builder.transform(X_hidden[va] if use_hidden else None, val_blocks)
        X_test = builder.transform(X_hidden[te] if use_hidden else None, test_blocks)
        names = _feature_group_names(builder.hidden_pca_dim, block_feature_names, enabled_blocks)
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
            "final_feature_dim": X_train.shape[1],
            "hidden_pca_dim": builder.hidden_pca_dim,
            "best_threshold": best_t,
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
                _prediction_frame(fold, "val", va, y, val_scores, best_t),
                _prediction_frame(fold, "test", te, y, test_scores, best_t),
            ]
        )
        artifact = (model, X_val, y[va], best_t)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_frames, ignore_index=True).to_csv(predictions_dir / f"{experiment}.csv", index=False)
    summary = {
        "experiment": experiment,
        "alignment_range": alignment_range,
        "use_hidden": use_hidden,
        "use_trajectory": "trajectory" in enabled_blocks,
        "use_alignment": "alignment" in enabled_blocks,
        "use_directional_truth": "directional_truth" in enabled_blocks,
        "pca_components": pca_components if use_hidden else 0,
        "svm_c": svm_c,
        "trajectory_feature_dim": feature_blocks["trajectory"].shape[1] if "trajectory" in enabled_blocks else 0,
        "alignment_feature_dim": feature_blocks["alignment"].shape[1] if "alignment" in enabled_blocks else 0,
        "directional_truth_feature_dim": feature_blocks["directional_truth"].shape[1] if "directional_truth" in enabled_blocks else 0,
        "runtime_seconds_total": time.time() - t0,
        **_aggregate(rows),
    }
    return summary, rows, feature_names, artifact


def _apply_debug_subset(X_hidden, layer_vectors, truth_layers, y, splits, max_samples):
    if max_samples is None or max_samples >= len(y):
        return X_hidden, layer_vectors, truth_layers, y, splits
    keep = set(range(max_samples))
    remap = {old: new for new, old in enumerate(sorted(keep))}
    new_splits = []
    for tr, va, te in splits:
        tr_new = np.asarray([remap[int(i)] for i in tr if int(i) in keep], dtype=int)
        va_new = np.asarray([remap[int(i)] for i in va if int(i) in keep], dtype=int) if va is not None else None
        te_new = np.asarray([remap[int(i)] for i in te if int(i) in keep], dtype=int)
        if tr_new.size and va_new is not None and va_new.size and te_new.size:
            new_splits.append((tr_new, va_new, te_new))
    if not new_splits:
        raise ValueError("--max-samples-debug removed all usable folds.")
    return X_hidden[:max_samples], layer_vectors[:max_samples], truth_layers[:max_samples], y[:max_samples], new_splits


def _copy_trajectory_cache(output_dir: Path, token_window: int) -> None:
    target = output_dir / f"trajectory_cache_tail{token_window}_mean21_24.npz"
    source = ROOT / "results" / "trajectory_features" / f"trajectory_cache_tail{token_window}_mean21_24.npz"
    if not target.exists() and source.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"Copied trajectory cache from {source} to {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "truth_alignment_ablation")
    parser.add_argument("--alignment-ranges", nargs="+", default=["10-24", "1-24", "21-24"])
    parser.add_argument("--token-window", type=int, default=32)
    parser.add_argument("--num-thresholds", type=int, default=1000)
    parser.add_argument("--truth-vector-mode", default="mean21_24")
    parser.add_argument("--truth-pooling", default="answer_mean")
    parser.add_argument("--reference-column", default=None)
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument("--answer-column", default=None)
    parser.add_argument("--label-column", default=None)
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
    config["layer_mapping"] = "cached index 0 = transformer layer 1; hidden_states[1] = transformer layer 1 when embeddings are present"
    with (output_dir / "config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2)

    _copy_trajectory_cache(output_dir, args.token_window)
    X_hidden, layer_vectors, y, splits = load_or_extract_representations(output_dir, args.token_window)
    truth_layers = load_or_extract_truth_layer_vectors(
        output_dir,
        args.token_window,
        args.prompt_column,
        args.reference_column,
        args.label_column,
        args.max_samples_debug,
    )
    X_hidden, layer_vectors, truth_layers, y, splits = _apply_debug_subset(
        X_hidden, layer_vectors, truth_layers, y, splits, args.max_samples_debug
    )
    print(f"Samples: {len(y)}")
    print(f"Hidden baseline shape: {X_hidden.shape}")
    print(f"Model-answer layer vectors: {layer_vectors.shape}")
    print(f"Truth/reference layer vectors: {truth_layers.shape}")
    for i, (tr, va, te) in enumerate(splits, 1):
        print(f"Fold {i}: train={len(tr)} {_class_distribution(y, tr)} val={len(va)} {_class_distribution(y, va)} test={len(te)} {_class_distribution(y, te)}")

    ranges = list(dict.fromkeys(args.alignment_ranges))
    feature_names_by_experiment = {}
    summaries, fold_details, artifacts = [], {}, {}

    # Precompute blocks by range.
    blocks_by_range = {}
    names_by_range = {}
    for range_label in ranges:
        layer_range = _parse_range(range_label)
        truth = truth_vectors_from_layers(truth_layers, args.truth_vector_mode, layer_range)
        traj_X, traj_names = build_trajectory_feature_matrix(layer_vectors, layer_range)
        align_X, align_names = build_alignment_feature_matrix(layer_vectors, truth, layer_range)
        dir_X, dir_names = build_directional_truth_feature_matrix(layer_vectors, truth, layer_range)
        blocks_by_range[range_label] = {
            "trajectory": _safe_matrix(traj_X, f"trajectory {range_label}"),
            "alignment": _safe_matrix(align_X, f"alignment {range_label}"),
            "directional_truth": _safe_matrix(dir_X, f"directional_truth {range_label}"),
        }
        names_by_range[range_label] = {
            "trajectory": traj_names,
            "alignment": align_names,
            "directional_truth": dir_names,
        }
        print(
            f"Range {range_label}: trajectory={traj_X.shape}, "
            f"alignment={align_X.shape}, directional={dir_X.shape}"
        )

    experiment_specs = [("E0_baseline_only", "none", [], True)]
    for range_label in ranges:
        if range_label == "10-24":
            experiment_specs.extend(
                [
                    ("E1_trajectory_10_24", range_label, ["trajectory"], True),
                    ("E2_alignment_only_10_24", range_label, ["alignment"], False),
                    ("E3_baseline_plus_alignment_10_24", range_label, ["alignment"], True),
                    ("E4_baseline_plus_directional_truth_10_24", range_label, ["directional_truth"], True),
                    ("E5_baseline_plus_alignment_plus_directional_10_24", range_label, ["alignment", "directional_truth"], True),
                    ("E6_baseline_plus_trajectory_plus_alignment_plus_directional_10_24", range_label, ["trajectory", "alignment", "directional_truth"], True),
                ]
            )
        else:
            safe = range_label.replace("-", "_")
            experiment_specs.extend(
                [
                    (f"E_baseline_plus_alignment_{safe}", range_label, ["alignment"], True),
                    (f"E_baseline_plus_alignment_plus_directional_{safe}", range_label, ["alignment", "directional_truth"], True),
                    (f"E_baseline_plus_trajectory_plus_alignment_plus_directional_{safe}", range_label, ["trajectory", "alignment", "directional_truth"], True),
                ]
            )

    empty_blocks = {"trajectory": np.zeros((len(y), 0), dtype=np.float32), "alignment": np.zeros((len(y), 0), dtype=np.float32), "directional_truth": np.zeros((len(y), 0), dtype=np.float32)}
    empty_names = {"trajectory": [], "alignment": [], "directional_truth": []}
    for name, range_label, enabled, use_hidden in experiment_specs:
        blocks = empty_blocks if range_label == "none" else blocks_by_range[range_label]
        block_names = empty_names if range_label == "none" else names_by_range[range_label]
        print(f"\n{name}: range={range_label}, blocks={enabled}, use_hidden={use_hidden}")
        summary, rows, feature_names, artifact = evaluate_experiment(
            name,
            range_label,
            X_hidden,
            blocks,
            block_names,
            enabled,
            y,
            splits,
            args.pca_components,
            args.svm_c,
            args.seed,
            args.num_thresholds,
            predictions_dir,
            use_hidden=use_hidden,
        )
        summaries.append(summary)
        fold_details[name] = rows
        feature_names_by_experiment[name] = feature_names
        artifacts[name] = artifact
        print(
            f"  val_f1_tuned={summary['val_f1_tuned']:.4f} "
            f"test_f1_tuned={summary['test_f1_tuned']:.4f} "
            f"val_auc={summary['val_roc_auc']:.4f} test_auc={summary['test_roc_auc']:.4f}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "summary.json").open("w") as f:
        json.dump({"summary": summaries, "folds": fold_details}, f, indent=2)
    with (output_dir / "feature_names.json").open("w") as f:
        json.dump(feature_names_by_experiment, f, indent=2)

    if args.run_permutation_importance:
        candidate = args.importance_experiment
        if candidate is None:
            candidate = str(
                summary_df.loc[summary_df["experiment"] != "E0_baseline_only"]
                .sort_values("val_f1_tuned", ascending=False)
                .iloc[0]["experiment"]
            )
        print(f"\nPermutation importance experiment: {candidate}")
        model, X_val, y_val, threshold = artifacts[candidate]
        names = feature_names_by_experiment[candidate]
        imp = permutation_importance_with_threshold(
            model, X_val, y_val, threshold, names, args.importance_repeats, args.seed
        )
        imp.to_csv(output_dir / "permutation_importance_best_val.csv", index=False)
        for group, filename in (
            ("alignment", "permutation_importance_alignment_only.csv"),
            ("directional_truth", "permutation_importance_directional_truth_only.csv"),
        ):
            imp.loc[imp["feature_group"] == group].to_csv(output_dir / filename, index=False)
        imp.loc[imp["feature_group"] != "hidden_pca"].to_csv(
            output_dir / "permutation_importance_nonhidden_only.csv", index=False
        )

    leaderboard_cols = [
        "experiment",
        "alignment_range",
        "final_feature_dim",
        "val_f1_tuned",
        "test_f1_tuned",
        "val_roc_auc",
        "test_roc_auc",
        "best_threshold",
    ]
    print("\nTruth alignment ablation leaderboard")
    print(
        summary_df.sort_values("val_f1_tuned", ascending=False)[leaderboard_cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    best_val = summary_df.sort_values("val_f1_tuned", ascending=False).iloc[0]
    best_test = summary_df.sort_values("test_f1_tuned", ascending=False).iloc[0]
    e0 = summary_df.loc[summary_df["experiment"] == "E0_baseline_only"].iloc[0]
    print(f"\nBest by val_f1_tuned: {best_val['experiment']} ({best_val['val_f1_tuned']:.4f})")
    print(f"Best by test_f1_tuned: {best_test['experiment']} ({best_test['test_f1_tuned']:.4f})")
    print(
        "Delta vs E0_baseline_only: "
        f"val_f1_tuned={best_val['val_f1_tuned'] - e0['val_f1_tuned']:.4f}, "
        f"test_f1_tuned={best_val['test_f1_tuned'] - e0['test_f1_tuned']:.4f}, "
        f"test_roc_auc={best_val['test_roc_auc'] - e0['test_roc_auc']:.4f}"
    )
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
